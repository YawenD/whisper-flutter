#include "WhisperBridge.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <android/log.h>
#include "whisper.h"

#define TAG "WHISPER"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_init_bridge(const char* path) {
    LOGI("[WHISPER] whisper_init_bridge called. model path=%s", path);
    struct whisper_context_params cparams = whisper_context_default_params();
    void* ctx = whisper_init_from_file_with_params(path, cparams);
    if (ctx) {
        LOGI("[WHISPER] Context initialized successfully: %p", ctx);
    } else {
        LOGE("[WHISPER] ERROR: Context initialization failed");
    }
    return ctx;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_full_bridge(void* ctx, float* samples, int n) {
    LOGI("[WHISPER] whisper_full_bridge called with ctx=%p, n_samples=%d", ctx, n);
    if (!ctx) {
        LOGE("[WHISPER] ERROR: Context is null");
        return -1;
    }
    if (!samples) {
        LOGE("[WHISPER] ERROR: Samples buffer is null");
        return -1;
    }
    if (n <= 0) {
        LOGE("[WHISPER] ERROR: Invalid sample count: %d", n);
        return -1;
    }
    
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // tu peux ici activer ou désactiver des options :
    params.print_progress = false;
    params.translate = false;
    params.language = "auto";

    LOGI("[WHISPER] Starting whisper_full processing...");
    int result = whisper_full((struct whisper_context*)ctx, params, samples, n);
    LOGI("[WHISPER] whisper_full completed with result=%d", result);
    
    if (result == 0) {
        int num_segments = whisper_full_n_segments((struct whisper_context*)ctx);
        LOGI("[WHISPER] Transcription successful. Number of segments: %d", num_segments);
    } else {
        LOGE("[WHISPER] ERROR: whisper_full returned error code: %d", result);
    }
    
    return result;
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

