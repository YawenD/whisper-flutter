#include "WhisperBridge.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <vector>
#include <mutex>
#include <unordered_map>
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

namespace {

struct SharedContext {
    std::string modelPath;
    struct whisper_context* ctx = nullptr;
    int refCount = 0;
};

struct WhisperStreamSession {
    SharedContext* shared = nullptr;
    std::vector<float> samples;
};

static std::unordered_map<std::string, SharedContext*> gContextCache;
static std::mutex gContextMutex;

static whisper_full_params make_default_full_params() {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = "fr";
    params.no_context = true;
    params.single_segment = false;
    return params;
}

static SharedContext* acquire_shared_context(const char* path) {
    if (!path) {
        NSLog(@"❌ [WHISPER] acquire_shared_context called with null path");
        return nullptr;
    }

    std::string key(path);
    std::lock_guard<std::mutex> guard(gContextMutex);
    auto it = gContextCache.find(key);
    if (it != gContextCache.end()) {
        it->second->refCount += 1;
        NSLog(@"🔁 [WHISPER] Reusing cached context for %s (refCount=%d)", path, it->second->refCount);
        return it->second;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(path, cparams);
    if (!ctx) {
        NSLog(@"❌ [WHISPER] Failed to initialize context for %s", path);
        return nullptr;
    }

    auto* shared = new SharedContext();
    shared->modelPath = key;
    shared->ctx = ctx;
    shared->refCount = 1;
    gContextCache.emplace(shared->modelPath, shared);
    NSLog(@"✅ [WHISPER] Context created and cached for %s", path);
    return shared;
}

static void release_shared_context(SharedContext* shared) {
    if (!shared) {
        NSLog(@"⚠️ [WHISPER] release_shared_context called with null shared pointer");
        return;
    }

    std::lock_guard<std::mutex> guard(gContextMutex);
    shared->refCount -= 1;
    if (shared->refCount > 0) {
        NSLog(@"ℹ️ [WHISPER] Context for %s still in use (refCount=%d)", shared->modelPath.c_str(), shared->refCount);
        return;
    }

    auto it = gContextCache.find(shared->modelPath);
    if (it != gContextCache.end()) {
        gContextCache.erase(it);
    }

    NSLog(@"🧹 [WHISPER] Releasing context for %s", shared->modelPath.c_str());
    whisper_free(shared->ctx);
    delete shared;
}

} // namespace

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
    
    struct whisper_full_params params = make_default_full_params();

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

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void* whisper_stream_session_create(const char* model_path) {
    NSLog(@"🔵 [WHISPER] whisper_stream_session_create called with model=%s", model_path ? model_path : "(null)");
    SharedContext* shared = acquire_shared_context(model_path);
    if (!shared) {
        NSLog(@"❌ [WHISPER] Failed to create stream session because context loading failed");
        return nullptr;
    }

    auto* session = new WhisperStreamSession();
    session->shared = shared;
    session->samples.reserve(16000); // reserve ~1 second by default
    return session;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_stream_session_destroy(void* session) {
    NSLog(@"🔵 [WHISPER] whisper_stream_session_destroy called with session=%p", session);
    if (!session) {
        NSLog(@"⚠️ [WHISPER] Session pointer is null in destroy");
        return;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.clear();
    typed->samples.shrink_to_fit();
    release_shared_context(typed->shared);
    delete typed;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_add_pcm16(void* session, const int16_t* samples, int n_samples) {
    if (!session) {
        NSLog(@"❌ [WHISPER] add_pcm16 called with null session");
        return -1;
    }
    if (!samples) {
        NSLog(@"❌ [WHISPER] add_pcm16 called with null samples buffer");
        return -2;
    }
    if (n_samples <= 0) {
        NSLog(@"⚠️ [WHISPER] add_pcm16 called with non-positive sample count: %d", n_samples);
        return 0;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.reserve(typed->samples.size() + static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        typed->samples.push_back(static_cast<float>(samples[i]) / 32768.0f);
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_add_pcm_f32(void* session, const float* samples, int n_samples) {
    if (!session) {
        NSLog(@"❌ [WHISPER] add_pcm_f32 called with null session");
        return -1;
    }
    if (!samples) {
        NSLog(@"❌ [WHISPER] add_pcm_f32 called with null samples buffer");
        return -2;
    }
    if (n_samples <= 0) {
        NSLog(@"⚠️ [WHISPER] add_pcm_f32 called with non-positive sample count: %d", n_samples);
        return 0;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.reserve(typed->samples.size() + static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        typed->samples.push_back(samples[i]);
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_stream_session_transcribe(void* session) {
    NSLog(@"🔵 [WHISPER] whisper_stream_session_transcribe called with session=%p", session);
    if (!session) {
        NSLog(@"❌ [WHISPER] transcribe called with null session");
        return -1;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    if (!typed->shared || !typed->shared->ctx) {
        NSLog(@"❌ [WHISPER] transcribe called with invalid context");
        return -2;
    }

    if (typed->samples.empty()) {
        NSLog(@"⚠️ [WHISPER] transcribe called with empty sample buffer");
        return -3;
    }

    struct whisper_full_params params = make_default_full_params();
    int result = whisper_full(typed->shared->ctx, params, typed->samples.data(), static_cast<int>(typed->samples.size()));
    if (result == 0) {
        int segments = whisper_full_n_segments(typed->shared->ctx);
        NSLog(@"✅ [WHISPER] Streaming transcription succeeded with %d segments", segments);
    } else {
        NSLog(@"❌ [WHISPER] Streaming transcription failed with code %d", result);
    }
    return result;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_stream_session_get_result(void* session) {
    if (!session) {
        NSLog(@"❌ [WHISPER] get_result called with null session");
        return nullptr;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    if (!typed->shared) {
        NSLog(@"❌ [WHISPER] get_result called with missing shared context");
        return nullptr;
    }

    return whisper_get_result_bridge(typed->shared->ctx);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_stream_session_reset(void* session) {
    NSLog(@"🔵 [WHISPER] whisper_stream_session_reset called with session=%p", session);
    if (!session) {
        NSLog(@"⚠️ [WHISPER] reset called with null session");
        return;
    }

    auto* typed = static_cast<WhisperStreamSession*>(session);
    typed->samples.clear();
    typed->samples.shrink_to_fit();
}