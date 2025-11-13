#include "WhisperBridge.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <thread>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <map>
#include <mutex>
#include <unordered_map>
#include <android/log.h>
#include "whisper.h"

#define TAG "WHISPER"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Detect high-performance CPU cores (big cores) like the official example
static int getHighPerfCpuCount() {
    try {
        // Method 1: Use CPU frequencies (preferred)
        std::vector<int> frequencies;
        int cpuIndex = 0;
        
        while (true) {
            std::ostringstream path;
            path << "/sys/devices/system/cpu/cpu" << cpuIndex << "/cpufreq/cpuinfo_max_freq";
            std::ifstream freqFile(path.str());
            
            if (!freqFile.good()) {
                break; // No more CPUs
            }
            
            int freq = 0;
            freqFile >> freq;
            if (freq > 0) {
                frequencies.push_back(freq);
            }
            cpuIndex++;
        }
        
        if (!frequencies.empty()) {
            // Sort frequencies
            std::sort(frequencies.begin(), frequencies.end());
            
            // Count CPUs with frequency > minimum (big cores)
            int minFreq = frequencies[0];
            int bigCoreCount = 0;
            for (int freq : frequencies) {
                if (freq > minFreq) {
                    bigCoreCount++;
                }
            }
            
            if (bigCoreCount > 0) {
                LOGI("Detected %d big cores (frequency-based)", bigCoreCount);
                return std::max(2, bigCoreCount); // At least 2 threads
            }
        }
        
        // Method 2: Fallback - use CPU variant from /proc/cpuinfo
        std::ifstream cpuInfo("/proc/cpuinfo");
        if (cpuInfo.good()) {
            std::vector<int> variants;
            std::string line;
            
            while (std::getline(cpuInfo, line)) {
                if (line.find("CPU variant") != std::string::npos) {
                    size_t pos = line.find("0x");
                    if (pos != std::string::npos) {
                        std::string hexStr = line.substr(pos + 2);
                        int variant = 0;
                        std::istringstream(hexStr) >> std::hex >> variant;
                        variants.push_back(variant);
                    }
                }
            }
            
            if (!variants.empty()) {
                std::sort(variants.begin(), variants.end());
                int minVariant = variants[0];
                int bigCoreCount = 0;
                for (int variant : variants) {
                    if (variant == minVariant) {
                        bigCoreCount++;
                    }
                }
                
                if (bigCoreCount > 0) {
                    LOGI("Detected %d big cores (variant-based)", bigCoreCount);
                    return std::max(2, bigCoreCount);
                }
            }
        }
    } catch (...) {
        LOGE("Error detecting CPU cores");
    }
    
    // Fallback: use hardware_concurrency minus 4 (like official example)
    unsigned int totalCores = std::thread::hardware_concurrency();
    int result = (totalCores > 4) ? (totalCores - 4) : totalCores;
    result = std::max(2, result); // At least 2 threads
    LOGI("Using fallback: %d threads (total cores: %u)", result, totalCores);
    return result;
}

namespace {

// Tracks a whisper_context instance shared by multiple streaming sessions
struct SharedContext {
    std::string modelPath;
    struct whisper_context* ctx = nullptr;
    int refCount = 0;
};

// Accumulates audio samples for a streaming request and references SharedContext
struct WhisperStreamSession {
    SharedContext* shared = nullptr;
    std::vector<float> samples;
};

static std::unordered_map<std::string, SharedContext*> gContextCache;
static std::mutex gContextMutex;

// Build the default inference parameters we want to share
static whisper_full_params make_default_full_params() {
    struct whisper_full_params params =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.no_context = true;
    params.single_segment = false;
    params.language = "fr";
    params.n_threads = getHighPerfCpuCount();
    return params;
}

// Load or reuse a whisper_context for a given model path
static SharedContext* acquire_shared_context(const char* path) {
    if (!path) {
        LOGE("[WHISPER] acquire_shared_context called with null path");
        return nullptr;
    }

    std::string key(path);
    std::lock_guard<std::mutex> guard(gContextMutex);
    auto it = gContextCache.find(key);
    if (it != gContextCache.end()) {
        it->second->refCount += 1;
        LOGI("[WHISPER] Reusing cached context for %s (refCount=%d)", path,
             it->second->refCount);
        return it->second;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    struct whisper_context* ctx =
        whisper_init_from_file_with_params(path, cparams);
    if (!ctx) {
        LOGE("[WHISPER] Failed to initialize context for %s", path);
        return nullptr;
    }

    auto* shared = new SharedContext();
    shared->modelPath = key;
    shared->ctx = ctx;
    shared->refCount = 1;
    gContextCache.emplace(shared->modelPath, shared);
    LOGI("[WHISPER] Context created and cached for %s", path);
    return shared;
}

// Decrement the reference and destroy the cached context if unused
static void release_shared_context(SharedContext* shared) {
    if (!shared) {
        LOGE("[WHISPER] release_shared_context called with null pointer");
        return;
    }

    std::lock_guard<std::mutex> guard(gContextMutex);
    shared->refCount -= 1;
    if (shared->refCount > 0) {
        LOGI("[WHISPER] Context for %s still in use (refCount=%d)",
             shared->modelPath.c_str(), shared->refCount);
        return;
    }

    auto it = gContextCache.find(shared->modelPath);
    if (it != gContextCache.end()) {
        gContextCache.erase(it);
    }

    LOGI("[WHISPER] Releasing context for %s", shared->modelPath.c_str());
    whisper_free(shared->ctx);
    delete shared;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_init_bridge(const char* path) {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;  // ⚡ CPU optimisé
    void* ctx = whisper_init_from_file_with_params(path, cparams);
    return ctx;
}

static char* dup_cstr(const char* s) {
    size_t n = std::strlen(s);
    char* out = new char[n + 1];
    std::memcpy(out, s, n + 1);
    return out;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_transcribe_from_file_bridge(const char* model_path, const char* wav_path) {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) return dup_cstr("Error: failed to load model");

    // Read and parse WAV file
    std::ifstream file(wav_path, std::ios::binary);
    if (!file) {
        whisper_free(ctx);
        return dup_cstr("Error: could not open audio file");
    }

    // RIFF header
    char riff_id[4];
    uint32_t riff_size = 0;
    char wave_id[4];
    if (!file.read(riff_id, 4) || !file.read(reinterpret_cast<char*>(&riff_size), 4) || !file.read(wave_id, 4)) {
        whisper_free(ctx);
        return dup_cstr("Error: invalid WAV header");
    }

    if (std::strncmp(riff_id, "RIFF", 4) != 0 || std::strncmp(wave_id, "WAVE", 4) != 0) {
        whisper_free(ctx);
        return dup_cstr("Error: not a RIFF/WAVE file");
    }

    // Parse chunks
    bool fmt_found = false;
    bool data_found = false;
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
    std::streampos data_pos = 0;

    while (file && (!fmt_found || !data_found)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (!file.read(chunk_id, 4) || !file.read(reinterpret_cast<char*>(&chunk_size), 4)) {
            break;
        }

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            // fmt chunk
            if (chunk_size < 16) {
                whisper_free(ctx);
                return dup_cstr("Error: invalid fmt chunk");
            }
            if (!file.read(reinterpret_cast<char*>(&audio_format), 2) ||
                !file.read(reinterpret_cast<char*>(&num_channels), 2) ||
                !file.read(reinterpret_cast<char*>(&sample_rate), 4)) {
                whisper_free(ctx);
                return dup_cstr("Error: truncated fmt chunk");
            }
            // skip byteRate (4) + blockAlign (2)
            file.seekg(6, std::ios::cur);
            if (!file.read(reinterpret_cast<char*>(&bits_per_sample), 2)) {
                whisper_free(ctx);
                return dup_cstr("Error: truncated fmt chunk");
            }
            // skip any extra fmt bytes
            if (chunk_size > 16) {
                file.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
            fmt_found = true;
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_found = true;
            data_size = chunk_size;
            data_pos = file.tellg();
            file.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        } else {
            // skip other chunks
            file.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }
    }

    if (!fmt_found || !data_found) {
        whisper_free(ctx);
        return dup_cstr("Error: missing fmt or data chunk");
    }

    if (audio_format != 1 /* PCM */ || num_channels != 1 || sample_rate != 16000 || bits_per_sample != 16) {
        whisper_free(ctx);
        return dup_cstr("Error: WAV must be PCM16 mono @16kHz");
    }

    if (data_size < 2) {
        whisper_free(ctx);
        return dup_cstr("Error: empty audio");
    }

    // Read PCM16 samples
    file.clear();
    file.seekg(data_pos);
    std::vector<int16_t> pcm16(data_size / 2);
    if (!file.read(reinterpret_cast<char*>(pcm16.data()), data_size)) {
        whisper_free(ctx);
        return dup_cstr("Error: could not read audio data");
    }

    // Convert to float32
    std::vector<float> pcmf32;
    pcmf32.reserve(pcm16.size());
    for (int16_t s : pcm16) {
        pcmf32.push_back(static_cast<float>(s) / 32768.0f);
    }

    // Aggressive trim: remove leading/trailing low-energy samples
    if (!pcmf32.empty()) {
        const float thr = 0.02f;  // Higher threshold for more aggressive trimming
        size_t i0 = 0, i1 = pcmf32.size();
        // Trim from start
        while (i0 < i1 && std::fabs(pcmf32[i0]) < thr) ++i0;
        // Trim from end
        while (i1 > i0 && std::fabs(pcmf32[i1 - 1]) < thr) --i1;
        // Apply trim if significant reduction
        if (i0 > 0 || i1 < pcmf32.size()) {
            std::vector<float> tmp;
            tmp.reserve(i1 - i0);
            tmp.insert(tmp.end(), pcmf32.begin() + i0, pcmf32.begin() + i1);
            pcmf32.swap(tmp);
            LOGI("Trimmed audio: %zu -> %zu samples", pcm16.size(), pcmf32.size());
        }
    }

    // Configure Whisper with aggressive performance optimizations
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // Use high-performance cores detection (like official example)
    params.n_threads = getHighPerfCpuCount();
    
    // Critical performance settings
    params.n_max_text_ctx = 0;           // No text context (faster)
    params.no_context = true;             // No past transcription context
    params.no_timestamps = true;         // Disable timestamps
    params.token_timestamps = false;      // Disable token-level timestamps
    params.single_segment = true;         // Force single segment
    params.translate = false;             // No translation
    params.language = "fr";               // Force French (no detection)
    params.detect_language = false;       // Disable language detection
    
    // Experimental speed-up techniques (very aggressive for short phrases)
    params.audio_ctx = 256;               // Reduce audio context even more (default is ~1500)
    
    // Limit output length for short phrases (8 words = ~10-15 tokens)
    params.max_tokens = 20;               // Limit tokens per segment for short phrases
    
    // Temperature and thresholds
    params.temperature = 0.0f;            // Greedy decoding (fastest)
    params.no_speech_thold = 0.9f;        // Very high threshold to skip silence very fast
    params.entropy_thold = 3.0f;          // Higher entropy threshold (skip low-confidence)
    params.logprob_thold = -0.8f;         // Higher logprob threshold
    
    // Disable all printing/logging
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;

    LOGI("samples=%zu sr=16000", pcmf32.size());
    LOGI("---> whisper_full");

    if (whisper_full(ctx, params, pcmf32.data(), pcmf32.size()) != 0) {
        whisper_free(ctx);
        return dup_cstr("Error: failed to process audio");
    }

    LOGI("<--- whisper_full");

    // Get the text
    std::ostringstream result;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        result << whisper_full_get_segment_text(ctx, i);
    }

    whisper_free(ctx);
    
    // Return the text in dynamic memory
    std::string text = result.str();
    char* output = new char[text.size() + 1];
    std::strcpy(output, text.c_str());
    return output;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_full_bridge(void* ctx, float* samples, int n) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // Use high-performance cores detection (like official example)
    params.n_threads = getHighPerfCpuCount();
    
    // Critical performance settings
    params.n_max_text_ctx = 0;
    params.no_context = true;
    params.no_timestamps = true;
    params.token_timestamps = false;
    params.single_segment = true;
    params.translate = false;
    params.language = "fr";
    params.detect_language = false;
    
    // Experimental speed-up techniques (very aggressive for short phrases)
    params.audio_ctx = 256;               // Reduce audio context even more
    
    // Limit output length for short phrases
    params.max_tokens = 20;
    
    // Temperature and thresholds
    params.temperature = 0.0f;
    params.no_speech_thold = 0.9f;        // Very high threshold
    params.entropy_thold = 3.0f;          // Higher entropy threshold
    params.logprob_thold = -0.8f;         // Higher logprob threshold
    
    // Disable all printing/logging
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    
    return whisper_full((struct whisper_context*)ctx, params, samples, n);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_bridge(void* ctx) {
    LOGI("[WHISPER] whisper_free_bridge called with ctx=%p", ctx);
    whisper_free((struct whisper_context*)ctx);
    LOGI("[WHISPER] Whisper context freed");
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_version_bridge() {
    return whisper_version();
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void enforce_binding(void) {
    // no-op; referenced to keep this object file in the final binary
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_get_result_bridge(void* ctx) {
    LOGI("[WHISPER] whisper_get_result_bridge called with ctx=%p", ctx);
    if (!ctx) {
        LOGE("[WHISPER] ERROR: Context is null in whisper_get_result_bridge");
        return nullptr;
    }
    
    struct whisper_context* wctx = (struct whisper_context*)ctx;
    const int num_segments = whisper_full_n_segments(wctx);
    LOGI("[WHISPER] Number of segments retrieved: %d", num_segments);

    std::string joined;
    joined.reserve(4096);
    for (int i = 0; i < num_segments; ++i) {
        const char* seg = whisper_full_get_segment_text(wctx, i);
        if (seg) {
            LOGI("[WHISPER] Segment %d: %s", i, seg);
            if (!joined.empty()) joined += " ";
            joined += seg;
        } else {
            LOGI("[WHISPER] WARNING: Segment %d text is null", i);
        }
    }

    LOGI("[WHISPER] Joined transcript length: %zu characters", joined.size());
    
    char* out = (char*)malloc(joined.size() + 1);
    if (!out) {
        LOGE("[WHISPER] ERROR: Failed to allocate memory for transcript");
        return nullptr;
    }
    std::memcpy(out, joined.c_str(), joined.size() + 1);
    LOGI("[WHISPER] Final transcript: %s", out);
    return out;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_cstr_bridge(const char* s) {
    if (s) free((void*)s);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_stream_session_create(const char* model_path) {
    LOGI("[WHISPER] whisper_stream_session_create called with model=%s",
         model_path ? model_path : "(null)");
    // Each session reuses (and increments) the shared context for this model
    SharedContext* shared = acquire_shared_context(model_path);
    if (!shared) {
        LOGE("[WHISPER] Failed to create stream session because context loading failed");
        return nullptr;
    }

    auto* session = new WhisperStreamSession();
    session->shared = shared;
    session->samples.reserve(16000);
    return session;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_stream_session_destroy(void* session) {
    LOGI("[WHISPER] whisper_stream_session_destroy called with session=%p", session);
    if (!session) {
        LOGE("[WHISPER] Session pointer is null in destroy");
        return;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.clear();
    typed->samples.shrink_to_fit();
    // Release the cached whisper context
    release_shared_context(typed->shared);
    delete typed;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_add_pcm16(void* session,
                                     const int16_t* samples,
                                     int n_samples) {
    // Append PCM16 data converted to float
    if (!session) {
        LOGE("[WHISPER] add_pcm16 called with null session");
        return -1;
    }
    if (!samples) {
        LOGE("[WHISPER] add_pcm16 called with null samples buffer");
        return -2;
    }
    if (n_samples <= 0) {
        LOGI("[WHISPER] add_pcm16 called with non-positive sample count: %d",
             n_samples);
        return 0;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.reserve(
        typed->samples.size() + static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        typed->samples.push_back(static_cast<float>(samples[i]) / 32768.0f);
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_add_pcm_f32(void* session,
                                       const float* samples,
                                       int n_samples) {
    // Append PCM32 data (already float)
    if (!session) {
        LOGE("[WHISPER] add_pcm_f32 called with null session");
        return -1;
    }
    if (!samples) {
        LOGE("[WHISPER] add_pcm_f32 called with null samples buffer");
        return -2;
    }
    if (n_samples <= 0) {
        LOGI("[WHISPER] add_pcm_f32 called with non-positive sample count: %d",
             n_samples);
        return 0;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.reserve(
        typed->samples.size() + static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        typed->samples.push_back(samples[i]);
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_transcribe(void* session) {
    LOGI("[WHISPER] whisper_stream_session_transcribe called with session=%p",
         session);
    // Run Whisper inference with the cached context and accumulated audio
    if (!session) {
        LOGE("[WHISPER] transcribe called with null session");
        return -1;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    if (!typed->shared || !typed->shared->ctx) {
        LOGE("[WHISPER] transcribe called with invalid context");
        return -2;
    }

    if (typed->samples.empty()) {
        LOGI("[WHISPER] transcribe called with empty sample buffer");
        return -3;
    }

    struct whisper_full_params params = make_default_full_params();
    int result = whisper_full(typed->shared->ctx, params,
                              typed->samples.data(),
                              static_cast<int>(typed->samples.size()));
    if (result == 0) {
        int segments = whisper_full_n_segments(typed->shared->ctx);
        LOGI("[WHISPER] Streaming transcription succeeded with %d segments",
             segments);
    } else {
        LOGE("[WHISPER] Streaming transcription failed with code %d", result);
    }
    return result;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_stream_session_get_result(void* session) {
    // Retrieve the merged text produced by the last transcription
    if (!session) {
        LOGE("[WHISPER] get_result called with null session");
        return nullptr;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    if (!typed->shared) {
        LOGE("[WHISPER] get_result called with missing shared context");
        return nullptr;
    }

    return whisper_get_result_bridge(typed->shared->ctx);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_stream_session_reset(void* session) {
    LOGI("[WHISPER] whisper_stream_session_reset called with session=%p",
         session);
    // Clear buffered samples so the next chunk starts clean
    if (!session) {
        LOGI("[WHISPER] reset called with null session");
        return;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.clear();
    typed->samples.shrink_to_fit();
}

