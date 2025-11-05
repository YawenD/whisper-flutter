#include "WhisperBridge.h"
#include <string>
#include <cstring>
#include <cstdio>
#import <Foundation/Foundation.h>
#import <os/log.h>
#if __has_feature(modules)
@import whisper;
#else
#  if __has_include(<whisper/whisper.h>)
#    include <whisper/whisper.h>
#  elif __has_include(<whisper.h>)
#    include <whisper.h>
#  else
#    error "whisper headers not found. Check pod integration."
#  endif
#endif

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_init_bridge(const char* path) {
    printf("[WHISPER] whisper_init_bridge called. model path=%s\n", path);
    NSLog(@"🔵 [WHISPER] whisper_init_bridge called. model path=%s", path);
    struct whisper_context_params cparams = whisper_context_default_params();
    void* ctx = whisper_init_from_file_with_params(path, cparams);
    if (ctx) {
        printf("[WHISPER] Context initialized successfully: %p\n", ctx);
        NSLog(@"✅ [WHISPER] Context initialized successfully: %p", ctx);
    } else {
        printf("[WHISPER] ERROR: Context initialization failed\n");
        NSLog(@"❌ [WHISPER] ERROR: Context initialization failed");
    }
    return ctx;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_full_bridge(void* ctx, float* samples, int n) {
    printf("[WHISPER] whisper_full_bridge called with ctx=%p, n_samples=%d\n", ctx, n);
    NSLog(@"🔵 [WHISPER] whisper_full_bridge called with ctx=%p, n_samples=%d", ctx, n);
    if (!ctx) {
        printf("[WHISPER] ERROR: Context is null\n");
        NSLog(@"❌ [WHISPER] ERROR: Context is null");
        return -1;
    }
    if (!samples) {
        printf("[WHISPER] ERROR: Samples buffer is null\n");
        NSLog(@"❌ [WHISPER] ERROR: Samples buffer is null");
        return -1;
    }
    if (n <= 0) {
        printf("[WHISPER] ERROR: Invalid sample count: %d\n", n);
        NSLog(@"❌ [WHISPER] ERROR: Invalid sample count: %d", n);
        return -1;
    }
    
    // Use default Whisper parameters (like the fast example)
    // The defaults are already optimized by Whisper creators
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = "fr";
    params.no_context = true;
    params.single_segment = false;

    printf("[WHISPER] Starting whisper_full processing...\n");
    NSLog(@"⏳ [WHISPER] Starting whisper_full processing...");
    int result = whisper_full((struct whisper_context*)ctx, params, samples, n);
    printf("[WHISPER] whisper_full completed with result=%d\n", result);
    NSLog(@"⏳ [WHISPER] whisper_full completed with result=%d", result);
    
    if (result == 0) {
        int num_segments = whisper_full_n_segments((struct whisper_context*)ctx);
        printf("[WHISPER] Transcription successful. Number of segments: %d\n", num_segments);
        NSLog(@"✅ [WHISPER] Transcription successful. Number of segments: %d", num_segments);
    } else {
        printf("[WHISPER] ERROR: whisper_full returned error code: %d\n", result);
        NSLog(@"❌ [WHISPER] ERROR: whisper_full returned error code: %d", result);
    }
    
    return result;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_bridge(void* ctx) {
    printf("[WHISPER] whisper_free_bridge called with ctx=%p\n", ctx);
    NSLog(@"🔵 [WHISPER] whisper_free_bridge called with ctx=%p", ctx);
    whisper_free((struct whisper_context*)ctx);
    printf("[WHISPER] Whisper context freed\n");
    NSLog(@"✅ [WHISPER] Whisper context freed");
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_version_bridge() {
    return whisper_version();
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void enforce_binding(void) {
    // no-op; referenced from Swift to keep this object file in the final binary
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_get_result_bridge(void* ctx) {
    printf("[WHISPER] whisper_get_result_bridge called with ctx=%p\n", ctx);
    NSLog(@"🔵 [WHISPER] whisper_get_result_bridge called with ctx=%p", ctx);
    if (!ctx) {
        printf("[WHISPER] ERROR: Context is null in whisper_get_result_bridge\n");
        NSLog(@"❌ [WHISPER] ERROR: Context is null in whisper_get_result_bridge");
        return nullptr;
    }
    
    struct whisper_context* wctx = (struct whisper_context*)ctx;
    const int num_segments = whisper_full_n_segments(wctx);
    printf("[WHISPER] Number of segments retrieved: %d\n", num_segments);
    NSLog(@"📊 [WHISPER] Number of segments retrieved: %d", num_segments);

    std::string joined;
    joined.reserve(4096);
    for (int i = 0; i < num_segments; ++i) {
        const char* seg = whisper_full_get_segment_text(wctx, i);
        if (seg) {
            printf("[WHISPER] Segment %d: %s\n", i, seg);
            NSLog(@"📝 [WHISPER] Segment %d: %s", i, seg);
            if (!joined.empty()) joined += " ";
            joined += seg;
        } else {
            printf("[WHISPER] WARNING: Segment %d text is null\n", i);
            NSLog(@"⚠️ [WHISPER] WARNING: Segment %d text is null", i);
        }
    }

    printf("[WHISPER] Joined transcript length: %zu characters\n", joined.size());
    NSLog(@"📏 [WHISPER] Joined transcript length: %zu characters", joined.size());
    
    char* out = (char*)malloc(joined.size() + 1);
    if (!out) {
        printf("[WHISPER] ERROR: Failed to allocate memory for transcript\n");
        NSLog(@"❌ [WHISPER] ERROR: Failed to allocate memory for transcript");
        return nullptr;
    }
    std::memcpy(out, joined.c_str(), joined.size() + 1);
    printf("[WHISPER] Final transcript: %s\n", out);
    NSLog(@"✅ [WHISPER] Final transcript: %s", out);
    return out;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_cstr_bridge(const char* s) {
    if (s) free((void*)s);
}