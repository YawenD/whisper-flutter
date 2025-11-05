#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Ces fonctions sont visibles depuis Dart via FFI (DynamicLibrary.process()).
// Elles sont implémentées dans WhisperBridge.mm

void* whisper_init_bridge(const char* model_path);
int whisper_full_bridge(void* ctx, float* samples, int n);
void whisper_free_bridge(void* ctx);
const char* whisper_version_bridge(void);

// Enforce bundling of this object file to prevent tree-shaking when using FFI
void enforce_binding(void);

const char* whisper_get_result_bridge(void* ctx);
void whisper_free_cstr_bridge(const char* s);

#ifdef __cplusplus
} // extern "C"
#endif