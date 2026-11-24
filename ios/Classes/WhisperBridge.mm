#include "WhisperBridge.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cmath>
#import <Foundation/Foundation.h>
#include <TargetConditionals.h>
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

// Encoder frames (50 per second) added after the audio with dynamic_audio_ctx.
static constexpr int kDefaultAudioCtxMargin = 128;

namespace {

struct SharedContext {
    std::string modelPath;
    struct whisper_context* ctx = nullptr;
    int refCount = 0;
    // whisper_full mutates the context, so sessions sharing one context must
    // serialize inference (and reading segments) through this mutex.
    std::mutex inferenceMutex;
};

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

// Metal on real devices; CPU on the iOS simulator, where ggml's Metal backend
// traps (SIGTRAP) right after compiling its shader library.
static whisper_context_params make_context_params() {
    struct whisper_context_params cparams = whisper_context_default_params();
#if TARGET_OS_SIMULATOR
    cparams.use_gpu = false;
#endif
    return cparams;
}

static whisper_full_params make_default_full_params(const char* language) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = (language && language[0] != '\0') ? language : "fr";
    params.no_context = true;
    params.single_segment = false;
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
// runaway repetition loops ("je je je ...") from burning seconds of compute.
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

    struct whisper_context_params cparams = make_context_params();
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
    NSLog(@"🔵 [WHISPER] whisper_init_bridge called. model path=%s", path);
    struct whisper_context_params cparams = make_context_params();
    void* ctx = whisper_init_from_file_with_params(path, cparams);
    if (ctx) {
        NSLog(@"✅ [WHISPER] Context initialized successfully: %p", ctx);
    } else {
        NSLog(@"❌ [WHISPER] ERROR: Context initialization failed");
    }
    return ctx;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int whisper_full_bridge(void* ctx, float* samples, int n) {
    NSLog(@"🔵 [WHISPER] whisper_full_bridge called with ctx=%p, n_samples=%d", ctx, n);
    if (!ctx) {
        NSLog(@"❌ [WHISPER] ERROR: Context is null");
        return -1;
    }
    if (!samples) {
        NSLog(@"❌ [WHISPER] ERROR: Samples buffer is null");
        return -1;
    }
    if (n <= 0) {
        NSLog(@"❌ [WHISPER] ERROR: Invalid sample count: %d", n);
        return -1;
    }

    struct whisper_full_params params = make_default_full_params(nullptr);
    int result = whisper_full((struct whisper_context*)ctx, params, samples, n);

    if (result == 0) {
        int num_segments = whisper_full_n_segments((struct whisper_context*)ctx);
        NSLog(@"✅ [WHISPER] Transcription successful. Number of segments: %d", num_segments);
    } else {
        NSLog(@"❌ [WHISPER] ERROR: whisper_full returned error code: %d", result);
    }

    return result;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void whisper_free_bridge(void* ctx) {
    NSLog(@"🔵 [WHISPER] whisper_free_bridge called with ctx=%p", ctx);
    whisper_free((struct whisper_context*)ctx);
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
    if (!ctx) {
        NSLog(@"❌ [WHISPER] ERROR: Context is null in whisper_get_result_bridge");
        return nullptr;
    }

    // Never log the transcript itself: user speech must not end up in system logs.
    std::string joined = collect_segments((struct whisper_context*)ctx);
    NSLog(@"📏 [WHISPER] Joined transcript length: %zu characters", joined.size());

    char* out = (char*)malloc(joined.size() + 1);
    if (!out) {
        NSLog(@"❌ [WHISPER] ERROR: Failed to allocate memory for transcript");
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
    NSLog(@"🔵 [WHISPER] whisper_stream_session_create called with model=%s", model_path ? model_path : "(null)");
    SharedContext* shared = acquire_shared_context(model_path);
    if (!shared) {
        NSLog(@"❌ [WHISPER] Failed to create stream session because context loading failed");
        return nullptr;
    }

    if (language && language[0] != '\0' && std::strcmp(language, "auto") != 0 &&
        whisper_lang_id(language) < 0) {
        NSLog(@"❌ [WHISPER] Unknown language code: %s", language);
        release_shared_context(shared);
        return nullptr;
    }

    auto* session = new WhisperStreamSession();
    session->shared = shared;
    if (language && language[0] != '\0') {
        session->language = language;
    }
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

    struct whisper_full_params params = make_default_full_params(typed->language.c_str());
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
    int result = whisper_full(typed->shared->ctx, params, typed->samples.data(), static_cast<int>(typed->samples.size()));
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
        NSLog(@"✅ [WHISPER] Streaming transcription succeeded with %d segments", whisper_full_n_segments(typed->shared->ctx));
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
    if (!typed->hasResult) {
        NSLog(@"❌ [WHISPER] get_result called before a successful transcribe");
        return nullptr;
    }

    char* out = (char*)malloc(typed->result.size() + 1);
    if (!out) {
        NSLog(@"❌ [WHISPER] ERROR: Failed to allocate memory for transcript");
        return nullptr;
    }
    std::memcpy(out, typed->result.c_str(), typed->result.size() + 1);
    return out;
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
    // Every arm64 Apple device supports the instructions the xcframework uses.
    return 1;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char* whisper_bridge_system_info(void) {
    return whisper_print_system_info();
}
