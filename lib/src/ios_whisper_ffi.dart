import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

typedef _WhisperStreamSessionCreateNative =
    Pointer<Void> Function(Pointer<Utf8>);
typedef _WhisperStreamSessionCreate = Pointer<Void> Function(Pointer<Utf8>);

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
        'Impossible de charger la bibliothèque native Whisper sur Android.',
      );
    }

    throw UnsupportedError(
      'WhisperNativeFfi ne prend pas en charge ${Platform.operatingSystem}.',
    );
  }

  late final _WhisperStreamSessionCreate _createSession =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionCreateNative>>(
            'whisper_stream_session_create',
          )
          .asFunction();

  late final _WhisperStreamSessionDestroy _destroySession =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionDestroyNative>>(
            'whisper_stream_session_destroy',
          )
          .asFunction();

  late final _WhisperStreamSessionAddPcm16 _addPcm16 =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionAddPcm16Native>>(
            'whisper_stream_session_add_pcm16',
          )
          .asFunction();

  late final _WhisperStreamSessionAddPcmF32 _addPcmF32 =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionAddPcmF32Native>>(
            'whisper_stream_session_add_pcm_f32',
          )
          .asFunction();

  late final _WhisperStreamSessionTranscribe _transcribe =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionTranscribeNative>>(
            'whisper_stream_session_transcribe',
          )
          .asFunction();

  late final _WhisperStreamSessionGetResult _getResult =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionGetResultNative>>(
            'whisper_stream_session_get_result',
          )
          .asFunction();

  late final _WhisperStreamSessionReset _resetSession =
      _library
          .lookup<NativeFunction<_WhisperStreamSessionResetNative>>(
            'whisper_stream_session_reset',
          )
          .asFunction();

  late final _WhisperFreeCString _freeCString =
      _library
          .lookup<NativeFunction<_WhisperFreeCStringNative>>(
            'whisper_free_cstr_bridge',
          )
          .asFunction();

  /// Create a new streaming session for the given model path.
  Pointer<Void> createSession(String modelPath) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      return _createSession(pathPtr);
    } finally {
      calloc.free(pathPtr);
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
      return ptr.toDartString();
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
  static WhisperStreamSession create(String modelPath) {
    final ffi = WhisperNativeFfi.instance;
    final handle = ffi.createSession(modelPath);
    if (handle == nullptr) {
      throw Exception('Unable to allocate Whisper streaming session');
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

    final view = Uint8List.fromList(bytes);
    final evenLength = view.lengthInBytes & ~1;
    if (evenLength == 0) {
      return;
    }

    final sampleCount = evenLength ~/ 2;
    final pointer = calloc<Int16>(sampleCount);
    try {
      final buffer = pointer.asTypedList(sampleCount);
      final byteData = ByteData.sublistView(view, 0, evenLength);
      for (var i = 0; i < sampleCount; i += 1) {
        buffer[i] = byteData.getInt16(i * 2, Endian.little);
      }
      _ffi.addPcm16(_handle, pointer, sampleCount);
    } finally {
      calloc.free(pointer);
    }
  }

  /// Enqueue floating-point samples from the VAD segmentation.
  void appendFloatSamples(List<double> samples) {
    if (_isDisposed) {
      throw StateError('Session already disposed');
    }
    if (samples.isEmpty) {
      return;
    }

    final length = samples.length;
    final pointer = calloc<Float>(length);
    try {
      final buffer = pointer.asTypedList(length);
      for (var i = 0; i < length; i += 1) {
        buffer[i] = samples[i].toDouble();
      }
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
