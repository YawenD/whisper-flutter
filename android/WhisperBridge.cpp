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
#include <atomic>
#include <chrono>
#include <android/log.h>
#include <sys/auxv.h>
#if defined(__aarch64__)
#include <asm/hwcap.h>
#endif
#include "whisper.h"

#define TAG "WHISPER"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::atomic<int> gCachedThreadCount{0};

// Detect high-performance CPU cores (big cores) like the official example
static int getHighPerfCpuCount() {
    int cached = gCachedThreadCount.load();
    if (cached > 0) {
        return cached;
    }

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
                int result = std::max(2, bigCoreCount);
                gCachedThreadCount.store(result);
                LOGI("Detected %d big cores (frequency-based)", bigCoreCount);
                return result; // At least 2 threads
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
                    int result = std::max(2, bigCoreCount);
                    gCachedThreadCount.store(result);
                    LOGI("Detected %d big cores (variant-based)", bigCoreCount);
                    return result;
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
    gCachedThreadCount.store(result);
    return result;
}

// Encoder frames (50 per second) added after the audio with dynamic_audio_ctx.
static constexpr int kDefaultAudioCtxMargin = 128;

namespace {

// Tracks a whisper_context instance shared by multiple streaming sessions
struct SharedContext {
    std::string modelPath;
    struct whisper_context* ctx = nullptr;
    int refCount = 0;
    // whisper_full mutates the context, so sessions sharing one context must
    // serialize inference (and reading segments) through this mutex.
    std::mutex inferenceMutex;
};

// Accumulates audio samples for a streaming request and references SharedContext
struct WhisperStreamSession {
    SharedContext* shared = nullptr;
    std::vector<float> samples;
    std::string language = "fr";
    // Segments are copied here right after whisper_full, while the inference
    // mutex is still held, so results from concurrent sessions cannot mix.
    std::string result;
    bool hasResult = false;
    whisper_bridge_stats stats{};
    bool hasStats = false;
    // Off by default: a shrunk encoder context measurably hurts accuracy.
    bool dynamicAudioCtx = false;
    std::string initialPrompt;
    int audioCtxMargin = kDefaultAudioCtxMargin;
    float temperatureInc = 0.0f;
    int bestOf = 1;
    int nThreads = 0;  // 0 = platform default
};

static std::unordered_map<std::string, SharedContext*> gContextCache;
static std::mutex gContextMutex;

// Build the default inference parameters we want to share
static whisper_full_params make_default_full_params(const char* language) {
    struct whisper_full_params params =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.no_context = true;
    params.single_segment = false;
    params.language = (language && language[0] != '\0') ? language : "fr";
    params.n_threads = getHighPerfCpuCount();
    // Disable the temperature fallback: when a greedy decode looks unreliable,
    // whisper.cpp would otherwise re-decode up to 5 times (0.2, 0.4, ...) with
    // best_of=5 candidates each, which causes multi-second latency spikes,
    // especially with small models that are often "unsure".
    params.temperature_inc = 0.0f;
    // Never emit non-speech tokens ([music], (applause), ...).
    params.suppress_nst = true;
    return params;
}

// Extra settings for short VAD segments (always < 30 s): a single segment
// without timestamps decodes fewer tokens, and capping the token count stops
// runaway repetition loops ("je je je ...") from burning seconds of CPU.
static void apply_short_segment_params(whisper_full_params& params,
                                       size_t n_samples,
                                       bool dynamic_audio_ctx,
                                       int audio_ctx_margin) {
    params.single_segment = true;
    params.no_timestamps = true;
    const float seconds = static_cast<float>(n_samples) / WHISPER_SAMPLE_RATE;
    // ~3-4 words/s in conversational French, ~1.5 tokens/word, plus margin.
    params.max_tokens = static_cast<int>(seconds * 8.0f) + 16;

    if (dynamic_audio_ctx) {
        // The encoder always processes a 30 s window (1500 frames, 50 per
        // second of audio) and attention cost grows quadratically with it: for
        // a 3 s phrase ~90% of the work is padding. Shrink the context to the
        // audio length plus a ~2.5 s margin (too tight a context makes the
        // model hallucinate), rounded up to a multiple of 64.
        int ctx = static_cast<int>(n_samples / 320) + audio_ctx_margin;
        ctx = ((ctx + 63) / 64) * 64;
        params.audio_ctx = std::min(1500, std::max(256, ctx));
    }
}

// The arm64 ggml kernels are compiled for armv8.2-a+dotprod+fp16 (see
// CMakeLists.txt). Older cores (Cortex-A53/A73 only) would SIGILL, so every
// entry point checks this first and fails cleanly instead.
static bool cpu_supports_build() {
#if defined(__aarch64__) && defined(WHISPER_REQUIRE_DOTPROD_FP16)
    static const bool supported = [] {
        const unsigned long hwcap = getauxval(AT_HWCAP);
        const bool ok = (hwcap & HWCAP_ASIMDDP) && (hwcap & HWCAP_ASIMDHP) &&
                        (hwcap & HWCAP_FPHP);
        if (!ok) {
            LOGE("[WHISPER] CPU lacks dotprod/fp16 (hwcap=0x%lx): unsupported",
                 hwcap);
        }
        return ok;
    }();
    return supported;
#else
    return true;
#endif
}

// Concatenate the segments of the last whisper_full run. The caller must hold
// the context's inference mutex.
static std::string collect_segments(struct whisper_context* ctx) {
    const int num_segments = whisper_full_n_segments(ctx);
    std::string joined;
    joined.reserve(4096);
    for (int i = 0; i < num_segments; ++i) {
        const char* seg = whisper_full_get_segment_text(ctx, i);
        if (seg) {
            if (!joined.empty()) joined += " ";
            joined += seg;
        }
    }
    return joined;
}

// Confidence of the last whisper_full run, used by the Dart side to drop
// hallucinations. The caller must hold the context's inference mutex.
static void collect_confidence(struct whisper_context* ctx,
                               whisper_bridge_stats& stats) {
    const whisper_token eot = whisper_token_eot(ctx);
    double sum_logprob = 0.0;
    int n_tokens = 0;
    float no_speech = 0.0f;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        no_speech = std::max(no_speech,
                             whisper_full_get_segment_no_speech_prob(ctx, i));
        const int n = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n; ++j) {
            const whisper_token_data data = whisper_full_get_token_data(ctx, i, j);
            if (data.id >= eot) {
                continue;  // special tokens (sot, language, timestamps...)
            }
            sum_logprob += data.plog;
            ++n_tokens;
        }
    }
    stats.n_tokens = n_tokens;
    stats.no_speech_prob = no_speech;
    stats.avg_logprob =
        n_tokens > 0 ? static_cast<float>(sum_logprob / n_tokens) : 0.0f;
}

// Load or reuse a whisper_context for a given model path
static SharedContext* acquire_shared_context(const char* path) {
    if (!path) {
        LOGE("[WHISPER] acquire_shared_context called with null path");
        return nullptr;
    }

    if (!cpu_supports_build()) {
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
    // Warm up number of threads so subsequent calls reuse cached value
    (void)getHighPerfCpuCount();
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
    if (!cpu_supports_build()) {
        return nullptr;
    }
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
    if (!cpu_supports_build()) {
        return dup_cstr("Error: unsupported CPU (dotprod/fp16 required)");
    }
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

    // Use the shared defaults so file-based and streaming transcriptions behave
    // the same way (the previous audio_ctx/max_tokens caps truncated anything
    // longer than a short phrase).
    whisper_full_params params = make_default_full_params(nullptr);

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
    struct whisper_full_params params = make_default_full_params(nullptr);
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
    if (!ctx) {
        LOGE("[WHISPER] ERROR: Context is null in whisper_get_result_bridge");
        return nullptr;
    }

    // Never log the transcript itself: user speech must not end up in logcat.
    std::string joined = collect_segments((struct whisper_context*)ctx);
    LOGI("[WHISPER] Joined transcript length: %zu characters", joined.size());

    char* out = (char*)malloc(joined.size() + 1);
    if (!out) {
        LOGE("[WHISPER] ERROR: Failed to allocate memory for transcript");
        return nullptr;
    }
    std::memcpy(out, joined.c_str(), joined.size() + 1);
    return out;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_cstr_bridge(const char* s) {
    if (s) free((void*)s);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_stream_session_create(const char* model_path, const char* language) {
    LOGI("[WHISPER] whisper_stream_session_create called with model=%s",
         model_path ? model_path : "(null)");
    // Each session reuses (and increments) the shared context for this model
    SharedContext* shared = acquire_shared_context(model_path);
    if (!shared) {
        LOGE("[WHISPER] Failed to create stream session because context loading failed");
        return nullptr;
    }

    if (language && language[0] != '\0' && std::strcmp(language, "auto") != 0 &&
        whisper_lang_id(language) < 0) {
        LOGE("[WHISPER] Unknown language code: %s", language);
        release_shared_context(shared);
        return nullptr;
    }

    auto* session = new WhisperStreamSession();
    session->shared = shared;
    if (language && language[0] != '\0') {
        session->language = language;
    }
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

    struct whisper_full_params params =
        make_default_full_params(typed->language.c_str());
    apply_short_segment_params(params, typed->samples.size(),
                               typed->dynamicAudioCtx, typed->audioCtxMargin);
    params.temperature_inc = typed->temperatureInc;
    params.greedy.best_of = typed->bestOf;
    if (typed->nThreads > 0) {
        params.n_threads = typed->nThreads;
    }
    if (!typed->initialPrompt.empty()) {
        params.initial_prompt = typed->initialPrompt.c_str();
    }

    // Hold the context mutex across inference AND segment collection so a
    // concurrent session on the same model cannot clobber the results.
    std::lock_guard<std::mutex> guard(typed->shared->inferenceMutex);
    whisper_reset_timings(typed->shared->ctx);
    const auto t_start = std::chrono::steady_clock::now();
    int result = whisper_full(typed->shared->ctx, params,
                              typed->samples.data(),
                              static_cast<int>(typed->samples.size()));
    const auto t_end = std::chrono::steady_clock::now();
    if (result == 0) {
        typed->result = collect_segments(typed->shared->ctx);
        typed->hasResult = true;

        whisper_bridge_stats stats{};
        stats.total_ms =
            std::chrono::duration<float, std::milli>(t_end - t_start).count();
        if (whisper_timings* t = whisper_get_timings(typed->shared->ctx)) {
            stats.sample_ms = t->sample_ms;
            stats.encode_ms = t->encode_ms;
            stats.decode_ms = t->decode_ms;
            stats.batchd_ms = t->batchd_ms;
            stats.prompt_ms = t->prompt_ms;
            delete t;  // allocated with new by whisper.cpp
        }
        stats.n_samples = static_cast<int32_t>(typed->samples.size());
        stats.n_threads = params.n_threads;
        stats.audio_ctx = params.audio_ctx > 0 ? params.audio_ctx : 1500;
        collect_confidence(typed->shared->ctx, stats);
        typed->stats = stats;
        typed->hasStats = true;
        LOGI("[WHISPER] Streaming transcription succeeded with %d segments",
             whisper_full_n_segments(typed->shared->ctx));
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
    if (!typed->hasResult) {
        LOGE("[WHISPER] get_result called before a successful transcribe");
        return nullptr;
    }

    char* out = (char*)malloc(typed->result.size() + 1);
    if (!out) {
        LOGE("[WHISPER] ERROR: Failed to allocate memory for transcript");
        return nullptr;
    }
    std::memcpy(out, typed->result.c_str(), typed->result.size() + 1);
    return out;
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
    typed->result.clear();
    typed->hasResult = false;
}


extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_get_stats(void* session, whisper_bridge_stats* out) {
    if (!session || !out) {
        return -1;
    }
    auto* typed = static_cast<WhisperStreamSession*>(session);
    if (!typed->hasStats) {
        return -2;
    }
    *out = typed->stats;
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_configure(void* session,
                                     const whisper_bridge_options* options) {
    if (!session || !options) {
        return -1;
    }
    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->dynamicAudioCtx = options->dynamic_audio_ctx != 0;
    typed->initialPrompt = options->initial_prompt ? options->initial_prompt : "";
    typed->audioCtxMargin = options->audio_ctx_margin > 0
                                ? options->audio_ctx_margin
                                : kDefaultAudioCtxMargin;
    typed->temperatureInc =
        options->temperature_inc >= 0.0f ? options->temperature_inc : 0.0f;
    typed->bestOf = options->best_of > 0 ? options->best_of : 1;
    typed->nThreads = options->n_threads > 0 ? options->n_threads : 0;
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_bridge_cpu_supported(void) {
    return cpu_supports_build() ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_bridge_system_info(void) {
    return whisper_print_system_info();
}
