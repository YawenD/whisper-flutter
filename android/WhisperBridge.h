#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// These functions are visible from Dart through FFI (DynamicLibrary.open('libwhisper*.so')).
// They are implemented in WhisperBridge.cpp.

void* whisper_init_bridge(const char* model_path);
int whisper_full_bridge(void* ctx, float* samples, int n);
void whisper_free_bridge(void* ctx);
const char* whisper_version_bridge(void);
const char* whisper_transcribe_from_file_bridge(const char* model_path, const char* wav_path);

// Enforce bundling of this object file to prevent tree-shaking when using FFI
void enforce_binding(void);

const char* whisper_get_result_bridge(void* ctx);
void whisper_free_cstr_bridge(const char* s);

// `language` is a whisper language code ("fr", "en", ...) or "auto"; NULL or empty
// falls back to "fr". Returns NULL when the language is unknown or loading fails.
void* whisper_stream_session_create(const char* model_path, const char* language);
void whisper_stream_session_destroy(void* session);
int whisper_stream_session_add_pcm16(void* session, const int16_t* samples, int n_samples);
int whisper_stream_session_add_pcm_f32(void* session, const float* samples, int n_samples);
int whisper_stream_session_transcribe(void* session);
const char* whisper_stream_session_get_result(void* session);
void whisper_stream_session_reset(void* session);

// Performance and confidence figures of the last successful
// whisper_stream_session_transcribe() call. Per-pass timings are averages
// reported by whisper.cpp (e.g. decode_ms is per token).
typedef struct whisper_bridge_stats {
    float total_ms;       // wall-clock time spent in whisper_full
    float sample_ms;      // per sampling step
    float encode_ms;      // per encoder pass
    float decode_ms;      // per decoded token
    float batchd_ms;      // per batched decode
    float prompt_ms;      // per prompt pass
    int32_t n_samples;    // number of 16 kHz samples fed to the model
    int32_t n_threads;    // CPU threads used by whisper_full
    int32_t audio_ctx;    // encoder context used (1500 = full 30 s window)
    int32_t n_tokens;     // decoded text tokens (special tokens excluded)
    float no_speech_prob; // highest no-speech probability over the segments
    float avg_logprob;    // mean log-probability of the text tokens
} whisper_bridge_stats;

// Copies the stats of the last transcription into `out`. Returns 0 on success.
int whisper_stream_session_get_stats(void* session, whisper_bridge_stats* out);

typedef struct whisper_bridge_options {
    // 1 = shrink the encoder context to the audio length (2-6x faster on short
    // segments but much less accurate: experimental), 0 = full 30 s window
    // (default).
    int32_t dynamic_audio_ctx;
    // Text prepended as decoder context (domain vocabulary, names...).
    // Copied by the callee; NULL or empty disables it.
    const char* initial_prompt;
    // Encoder frames added after the audio when dynamic_audio_ctx is on
    // (50 frames = 1 s). <= 0 uses the default (128).
    int32_t audio_ctx_margin;
    // Temperature step of the fallback re-decoding (0 = no fallback, the
    // default; e.g. 0.4 allows 2 retries at 0.4 and 0.8). Negative = default.
    float temperature_inc;
    // Candidates sampled per fallback retry. <= 0 uses the default (1).
    int32_t best_of;
    // CPU threads for whisper_full. <= 0 uses the platform default.
    int32_t n_threads;
} whisper_bridge_options;

// Applies `options` to the following transcriptions. Returns 0 on success.
int whisper_stream_session_configure(void* session, const whisper_bridge_options* options);

// Returns 1 when this CPU supports the instruction set the native library was
// built for, 0 otherwise (calling into whisper would then crash with SIGILL).
int whisper_bridge_cpu_supported(void);

// whisper.cpp / ggml build features (e.g. "NEON = 1 | DOTPROD = 1 | ...").
// Static string owned by the library: do NOT free it.
const char* whisper_bridge_system_info(void);

#ifdef __cplusplus
} // extern "C"
#endif

