#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// These functions are visible from Dart through FFI (DynamicLibrary.process()).
// They are implemented in WhisperBridge.mm.

void* whisper_init_bridge(const char* model_path);
int whisper_full_bridge(void* ctx, float* samples, int n);
void whisper_free_bridge(void* ctx);
const char* whisper_version_bridge(void);

// Enforce bundling of this object file to prevent tree-shaking when using FFI.
void enforce_binding(void);

const char* whisper_get_result_bridge(void* ctx);
void whisper_free_cstr_bridge(const char* s);

// Streaming helpers.
void* whisper_stream_session_create(const char* model_path);
void whisper_stream_session_destroy(void* session);
int whisper_stream_session_add_pcm16(void* session, const int16_t* samples, int n_samples);
int whisper_stream_session_add_pcm_f32(void* session, const float* samples, int n_samples);
int whisper_stream_session_transcribe(void* session);
const char* whisper_stream_session_get_result(void* session);
void whisper_stream_session_reset(void* session);

#ifdef __cplusplus
} // extern "C"
#endif
