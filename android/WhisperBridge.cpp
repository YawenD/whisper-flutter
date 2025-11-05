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
#include <android/log.h>
#include "whisper.h"

#define TAG "WHISPER"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

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

    // Quick trim: remove leading/trailing low-energy samples
    if (!pcmf32.empty()) {
        const float thr = 0.015f;
        size_t i0 = 0, i1 = pcmf32.size();
        while (i0 < i1 && std::fabs(pcmf32[i0]) < thr) ++i0;
        while (i1 > i0 && std::fabs(pcmf32[i1 - 1]) < thr) --i1;
        if (i0 > 0 || i1 < pcmf32.size()) {
            std::vector<float> tmp;
            tmp.reserve(i1 - i0);
            tmp.insert(tmp.end(), pcmf32.begin() + i0, pcmf32.begin() + i1);
            pcmf32.swap(tmp);
        }
    }

    // Configure Whisper
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 4;
    params.n_threads = (int) std::min(8u, cores);
    params.n_max_text_ctx = 0;
    params.no_context = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_timestamps = false;
    params.translate = false;
    params.language = "fr";

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
    LOGI("[WHISPER] whisper_full_bridge called with ctx=%p, n_samples=%d", ctx, n);
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 4;
    params.n_threads = std::min(4u, cores); // ✅ limite aux big cores
    params.language = "fr";
    params.translate = false;
    params.no_timestamps = true;
    params.token_timestamps = false; // ✅ souvent oublié
    params.temperature = 0.0f;
    params.n_max_text_ctx = 0;
    params.single_segment = true;
    params.print_progress = false;
    params.print_realtime = false;
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

