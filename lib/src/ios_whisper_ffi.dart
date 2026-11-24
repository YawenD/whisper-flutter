import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'whisper_stats.dart';

typedef _WhisperStreamSessionCreateNative =
    Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _WhisperStreamSessionCreate =
    Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);

typedef _WhisperStreamSessionDestroyNative = Void Function(Pointer<Void>);
typedef _WhisperStreamSessionDestroy = void Function(Pointer<Void>);

typedef _WhisperStreamSessionAddPcm16Native =
    Int32 Function(Pointer<Void>, Pointer<Int16>, Int32);
typedef _WhisperStreamSessionAddPcm16 =
    int Function(Pointer<Void>, Pointer<Int16>, int);

typedef _WhisperStreamSessionAddPcmF32Native =
    Int32 Function(Pointer<Void>, Pointer<Float>, Int32);
typedef _WhisperStreamSessionAddPcmF32 =
    int Function(Pointer<Void>, Pointer<Float>, int);

typedef _WhisperStreamSessionTranscribeNative = Int32 Function(Pointer<Void>);
typedef _WhisperStreamSessionTranscribe = int Function(Pointer<Void>);

typedef _WhisperStreamSessionGetResultNative =
    Pointer<Utf8> Function(Pointer<Void>);
typedef _WhisperStreamSessionGetResult = Pointer<Utf8> Function(Pointer<Void>);

typedef _WhisperStreamSessionResetNative = Void Function(Pointer<Void>);
typedef _WhisperStreamSessionReset = void Function(Pointer<Void>);

typedef _WhisperFreeCStringNative = Void Function(Pointer<Utf8>);
typedef _WhisperFreeCString = void Function(Pointer<Utf8>);

typedef _WhisperStreamSessionGetStatsNative =
    Int32 Function(Pointer<Void>, Pointer<_WhisperBridgeStats>);
typedef _WhisperStreamSessionGetStats =
    int Function(Pointer<Void>, Pointer<_WhisperBridgeStats>);

typedef _WhisperStreamSessionConfigureNative =
    Int32 Function(Pointer<Void>, Pointer<_WhisperBridgeOptions>);
typedef _WhisperStreamSessionConfigure =
    int Function(Pointer<Void>, Pointer<_WhisperBridgeOptions>);

typedef _WhisperCpuSupportedNative = Int32 Function();
typedef _WhisperCpuSupported = int Function();

typedef _WhisperSystemInfoNative = Pointer<Utf8> Function();
typedef _WhisperSystemInfo = Pointer<Utf8> Function();

/// Mirror of `whisper_bridge_stats` in WhisperBridge.h.
final class _WhisperBridgeStats extends Struct {
  @Float()
  external double totalMs;
  @Float()
  external double sampleMs;
  @Float()
  external double encodeMs;
  @Float()
  external double decodeMs;
  @Float()
  external double batchdMs;
  @Float()
  external double promptMs;
  @Int32()
  external int nSamples;
  @Int32()
  external int nThreads;
  @Int32()
  external int audioCtx;
  @Int32()
  external int nTokens;
  @Float()
  external double noSpeechProb;
  @Float()
  external double avgLogprob;
}

/// Mirror of `whisper_bridge_options` in WhisperBridge.h.
final class _WhisperBridgeOptions extends Struct {
  @Int32()
  external int dynamicAudioCtx;
  external Pointer<Utf8> initialPrompt;
  @Int32()
  external int audioCtxMargin;
  @Float()
  external double temperatureInc;
  @Int32()
  external int bestOf;
  @Int32()
  external int nThreads;
}

/// Decode a native NUL-terminated string, tolerating invalid UTF-8: whisper
/// emits text token by token and a multi-byte character (accents...) can be
/// split by the token cap, which would make a strict decoder throw.
String _decodeUtf8(Pointer<Utf8> ptr) {
  final bytes = ptr.cast<Uint8>();
  var length = 0;
  while (bytes[length] != 0) {
    length++;
  }
  return utf8.decode(bytes.asTypedList(length), allowMalformed: true);
}

class WhisperNativeFfi {
  WhisperNativeFfi._internal() : _library = _openLibrary() {
    try {
      _library
          .lookupFunction<Void Function(), void Function()>('enforce_binding')
          .call();
    } catch (_) {
      // Some builds may strip enforce_binding; ignore if missing.
    }
  }

  static final WhisperNativeFfi instance = WhisperNativeFfi._internal();

  final DynamicLibrary _library;

  /// Resolve the platform-specific Whisper dynamic library.
  static DynamicLibrary _openLibrary() {
    if (Platform.isIOS) {
      return DynamicLibrary.process();
    }

    if (Platform.isAndroid) {
      final candidates = <String>[
        'libwhisper_v8fp16_va.so',
        'libwhisper_vfpv4.so',
        'libwhisper.so',
      ];
      for (final name in candidates) {
        try {
          return DynamicLibrary.open(name);
        } catch (_) {
          // Try next candidate.
        }
      }
      throw UnsupportedError(
        'Unable to load the native Whisper library on Android.',
      );
    }

    throw UnsupportedError(
      'WhisperNativeFfi does not support ${Platform.operatingSystem}.',
    );
  }

  late final _WhisperStreamSessionCreate _createSession = _library
      .lookup<NativeFunction<_WhisperStreamSessionCreateNative>>(
        'whisper_stream_session_create',
      )
      .asFunction();

  late final _WhisperStreamSessionDestroy _destroySession = _library
      .lookup<NativeFunction<_WhisperStreamSessionDestroyNative>>(
        'whisper_stream_session_destroy',
      )
      .asFunction();

  late final _WhisperStreamSessionAddPcm16 _addPcm16 = _library
      .lookup<NativeFunction<_WhisperStreamSessionAddPcm16Native>>(
        'whisper_stream_session_add_pcm16',
      )
      .asFunction();

  late final _WhisperStreamSessionAddPcmF32 _addPcmF32 = _library
      .lookup<NativeFunction<_WhisperStreamSessionAddPcmF32Native>>(
        'whisper_stream_session_add_pcm_f32',
      )
      .asFunction();

  late final _WhisperStreamSessionTranscribe _transcribe = _library
      .lookup<NativeFunction<_WhisperStreamSessionTranscribeNative>>(
        'whisper_stream_session_transcribe',
      )
      .asFunction();

  late final _WhisperStreamSessionGetResult _getResult = _library
      .lookup<NativeFunction<_WhisperStreamSessionGetResultNative>>(
        'whisper_stream_session_get_result',
      )
      .asFunction();

  late final _WhisperStreamSessionReset _resetSession = _library
      .lookup<NativeFunction<_WhisperStreamSessionResetNative>>(
        'whisper_stream_session_reset',
      )
      .asFunction();

  late final _WhisperFreeCString _freeCString = _library
      .lookup<NativeFunction<_WhisperFreeCStringNative>>(
        'whisper_free_cstr_bridge',
      )
      .asFunction();

  late final _WhisperStreamSessionGetStats _getStats = _library
      .lookup<NativeFunction<_WhisperStreamSessionGetStatsNative>>(
        'whisper_stream_session_get_stats',
      )
      .asFunction();

  late final _WhisperStreamSessionConfigure _configure = _library
      .lookup<NativeFunction<_WhisperStreamSessionConfigureNative>>(
        'whisper_stream_session_configure',
      )
      .asFunction();

  late final _WhisperCpuSupported _cpuSupported = _library
      .lookup<NativeFunction<_WhisperCpuSupportedNative>>(
        'whisper_bridge_cpu_supported',
      )
      .asFunction();

  late final _WhisperSystemInfo _systemInfo = _library
      .lookup<NativeFunction<_WhisperSystemInfoNative>>(
        'whisper_bridge_system_info',
      )
      .asFunction();

  /// Whether this CPU supports the instruction set the native library was
  /// built for (always true on iOS).
  bool get isCpuSupported => _cpuSupported() == 1;

  /// whisper.cpp / ggml build features, e.g. "NEON = 1 | DOTPROD = 1 | ...".
  String get systemInfo => _decodeUtf8(_systemInfo());

  /// Stats of the last successful transcription of [session], if any.
  WhisperStats? lastStats(Pointer<Void> session) {
    final out = calloc<_WhisperBridgeStats>();
    try {
      if (_getStats(session, out) != 0) {
        return null;
      }
      final t = out.ref;
      return WhisperStats(
        totalMs: t.totalMs,
        sampleMs: t.sampleMs,
        encodeMs: t.encodeMs,
        decodeMs: t.decodeMs,
        batchdMs: t.batchdMs,
        promptMs: t.promptMs,
        sampleCount: t.nSamples,
        threadCount: t.nThreads,
        audioCtx: t.audioCtx,
        tokenCount: t.nTokens,
        noSpeechProb: t.noSpeechProb,
        avgLogprob: t.avgLogprob,
      );
    } finally {
      calloc.free(out);
    }
  }

  /// Apply decoding [options] to [session].
  void configure(Pointer<Void> session, WhisperDecodingOptions options) {
    final out = calloc<_WhisperBridgeOptions>();
    final prompt = options.initialPrompt;
    final promptPtr = (prompt == null || prompt.isEmpty)
        ? nullptr.cast<Utf8>()
        : prompt.toNativeUtf8();
    try {
      out.ref
        ..dynamicAudioCtx = options.dynamicAudioCtx ? 1 : 0
        ..initialPrompt = promptPtr
        ..audioCtxMargin = options.audioCtxMargin
        ..temperatureInc = options.temperatureInc
        ..bestOf = options.bestOf
        ..nThreads = options.threads;
      final code = _configure(session, out);
      if (code != 0) {
        throw Exception('Failed to configure Whisper session (code: $code)');
      }
    } finally {
      if (promptPtr != nullptr) calloc.free(promptPtr);
      calloc.free(out);
    }
  }

  /// Create a new streaming session for the given model path and language.
  Pointer<Void> createSession(String modelPath, {String language = 'fr'}) {
    final pathPtr = modelPath.toNativeUtf8();
    final languagePtr = language.toNativeUtf8();
    try {
      return _createSession(pathPtr, languagePtr);
    } finally {
      calloc.free(pathPtr);
      calloc.free(languagePtr);
    }
  }

  /// Destroy the native session and release shared resources.
  void destroySession(Pointer<Void> session) {
    _destroySession(session);
  }

  /// Reset buffered audio for the native session.
  void resetSession(Pointer<Void> session) {
    _resetSession(session);
  }

  /// Append PCM16 audio frames (converted to float on the native side).
  void addPcm16(
    Pointer<Void> session,
    Pointer<Int16> samples,
    int sampleCount,
  ) {
    final result = _addPcm16(session, samples, sampleCount);
    if (result != 0) {
      throw Exception('Failed to append PCM16 chunk (code: $result)');
    }
  }

  /// Append PCM32 audio frames for the current session.
  void addPcmF32(
    Pointer<Void> session,
    Pointer<Float> samples,
    int sampleCount,
  ) {
    final result = _addPcmF32(session, samples, sampleCount);
    if (result != 0) {
      throw Exception('Failed to append PCM32 chunk (code: $result)');
    }
  }

  /// Trigger a synchronous transcription on the buffered audio.
  int transcribe(Pointer<Void> session) {
    return _transcribe(session);
  }

  /// Retrieve and free the transcription string built by the native code.
  String takeResult(Pointer<Void> session) {
    final ptr = _getResult(session);
    if (ptr == nullptr) {
      throw Exception('No transcription result available');
    }
    try {
      return _decodeUtf8(ptr);
    } finally {
      _freeCString(ptr);
    }
  }
}

class WhisperStreamSession {
  WhisperStreamSession._(this._ffi, this._handle);

  final WhisperNativeFfi _ffi;
  final Pointer<Void> _handle;
  bool _isDisposed = false;

  /// Build a streaming session backed by the shared native context.
  static WhisperStreamSession create(String modelPath, {String language = 'fr'}) {
    final ffi = WhisperNativeFfi.instance;
    if (!ffi.isCpuSupported) {
      throw UnsupportedError(
        'This CPU lacks the ARMv8.2 dotprod/fp16 extensions required by the '
        'native Whisper library.',
      );
    }
    final handle = ffi.createSession(modelPath, language: language);
    if (handle == nullptr) {
      throw Exception(
        'Unable to create the Whisper session (model "$modelPath", language '
        '"$language"): the model file may be missing or corrupt, or the '
        'language code unknown.',
      );
    }
    return WhisperStreamSession._(ffi, handle);
  }

  /// Enqueue raw PCM16 audio bytes produced by the recorder.
  void appendPcmBytes(List<int> bytes) {
    if (_isDisposed) {
      throw StateError('Session already disposed');
    }
    if (bytes.isEmpty) {
      return;
    }

    final view = bytes is Uint8List ? bytes : Uint8List.fromList(bytes);
    final evenLength = view.lengthInBytes & ~1;
    if (evenLength == 0) {
      return;
    }

    final sampleCount = evenLength ~/ 2;
    final pointer = calloc<Int16>(sampleCount);
    try {
      // Raw byte copy: PCM16 from the recorder is little-endian, like every
      // Android/iOS CPU, so no per-sample decoding is needed.
      pointer
          .cast<Uint8>()
          .asTypedList(evenLength)
          .setRange(0, evenLength, view);
      _ffi.addPcm16(_handle, pointer, sampleCount);
    } finally {
      calloc.free(pointer);
    }
  }

  /// Enqueue floating-point samples from the VAD segmentation.
  void appendFloatSamples(Float32List samples) {
    if (_isDisposed) {
      throw StateError('Session already disposed');
    }
    if (samples.isEmpty) {
      return;
    }

    final length = samples.length;
    final pointer = calloc<Float>(length);
    try {
      pointer.asTypedList(length).setRange(0, length, samples);
      _ffi.addPcmF32(_handle, pointer, length);
    } finally {
      calloc.free(pointer);
    }
  }

  /// Run inference synchronously and return the transcript.
  String transcribeSync() {
    if (_isDisposed) {
      throw StateError('Session already disposed');
    }

    final code = _ffi.transcribe(_handle);
    if (code != 0) {
      throw Exception('Whisper transcription failed with code $code');
    }
    return _ffi.takeResult(_handle);
  }

  /// Stats of the last successful [transcribeSync] call, if any.
  WhisperStats? get lastStats => _isDisposed ? null : _ffi.lastStats(_handle);

  /// Apply decoding [options] to the following transcriptions.
  void configure(WhisperDecodingOptions options) {
    if (_isDisposed) {
      throw StateError('Session already disposed');
    }
    _ffi.configure(_handle, options);
  }

  /// Clear pending samples inside the native session.
  void reset() {
    if (_isDisposed) {
      return;
    }
    _ffi.resetSession(_handle);
  }

  /// Release native resources owned by this session.
  void dispose() {
    if (_isDisposed) {
      return;
    }
    _ffi.destroySession(_handle);
    _isDisposed = true;
  }
}
