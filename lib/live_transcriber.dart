import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_whisper_ggml/record_adapter.dart';
import 'package:flutter_whisper_ggml/src/ios_whisper_ffi.dart';
import 'package:flutter_whisper_ggml/vad_adapter.dart';

class WhisperLiveTranscriber {
  WhisperLiveTranscriber({
    WhisperMethodChannel? whisper,
    IRecordAdapter? recordAdapter,
    IVadAdapter? vadAdapter,
  }) : _methodChannel = whisper ?? WhisperMethodChannel(),
       _recordAdapter = recordAdapter ?? RecordAdapter(),
       _vadAdapter = vadAdapter ?? VadAdapter();

  final WhisperMethodChannel _methodChannel;
  final IRecordAdapter _recordAdapter;
  final IVadAdapter _vadAdapter;

  WhisperStreamSession? _whisperSession;
  StreamSubscription<Uint8List>? _recordSubscription;
  StreamController<String>? _transcriptsController;
  Future<void> _processing = Future.value();
  bool _listening = false;
  bool _vadReady = false;
  String? _modelPath;

  Stream<String> get transcripts {
    final controller = _transcriptsController;
    if (controller == null) {
      throw StateError(
        'Call startListening() before listening to the transcripts stream.',
      );
    }
    return controller.stream;
  }

  Future<void> prepare({
    required String modelAssetPath,
    String? modelPath,
  }) async {
    _modelPath = await _methodChannel.prepareModel(modelAssetPath);
    if (modelPath != null) {
      await _vadAdapter.initSilero(modelPath: modelPath);
      _vadReady = true;
    }
  }

  Future<void> initSession() async {
    if (_whisperSession != null) {
      return;
    }
    final modelPath = _modelPath;
    if (modelPath == null) {
      throw StateError(
        'prepare() must be called with a model path before initSession().',
      );
    }
    _whisperSession = WhisperStreamSession.create(modelPath);
  }

  Future<void> startListening() async {
    if (_listening) {
      throw StateError('Live transcription already started.');
    }
    final modelPath = _modelPath;
    if (modelPath == null) {
      throw StateError(
        'prepare() doit être appelé avec un chemin de modèle avant startListening().',
      );
    }

    _transcriptsController = StreamController<String>.broadcast();
    if (_whisperSession == null) {
      _whisperSession = WhisperStreamSession.create(modelPath);
    } else {
      _whisperSession!.reset();
    }
    final audioStream = await _recordAdapter.startStream();

    _recordSubscription = audioStream.listen(
      (chunk) {
        _processing = _processing.then((_) => _handleChunk(chunk));
      },
      onError: (Object error, StackTrace stackTrace) {
        _transcriptsController?.addError(error, stackTrace);
      },
      onDone: () async {
        await _processing;
        await _closeController();
      },
      cancelOnError: false,
    );

    _listening = true;
  }

  Future<void> stopListening() async {
    if (!_listening) return;
    _listening = false;

    await _recordSubscription?.cancel();
    _recordSubscription = null;
    await _recordAdapter.stopStream();

    await _processing;

    if (!_vadReady) {
      await _flushPendingTranscript();
    }

    await _closeController();
    _whisperSession?.dispose();
    _whisperSession = null;
  }

  Future<void> dispose() async {
    await stopListening();
    await _recordAdapter.dispose();
    await _vadAdapter.dispose();
  }

  Future<void> _handleChunk(Uint8List chunk) async {
    final session = _whisperSession;
    if (session == null) {
      return;
    }

    if (!_vadReady) {
      session.appendPcmBytes(chunk);
      return;
    }

    try {
      await for (final speechSamples in _vadAdapter.listenToSpeechEnd(chunk)) {
        if (speechSamples.isEmpty) {
          if (kDebugMode) {
            print('[WhisperLiveTranscriber] speechSamples is empty');
          }
          continue;
        }
        if (kDebugMode) {
          print(
            '[WhisperLiveTranscriber] speechSamples: ${speechSamples.length}',
          );
        }

        session.appendFloatSamples(speechSamples);
        final transcript = session.transcribeSync().trim();
        session.reset();

        if (transcript.isNotEmpty) {
          _transcriptsController?.add(transcript);
        }
      }
    } catch (error, stackTrace) {
      _transcriptsController?.addError(error, stackTrace);
    }
  }

  Future<void> _flushPendingTranscript() async {
    final session = _whisperSession;
    if (session == null) {
      return;
    }

    try {
      final transcript = session.transcribeSync().trim();
      if (transcript.isNotEmpty) {
        _transcriptsController?.add(transcript);
      }
    } catch (error, stackTrace) {
      _transcriptsController?.addError(error, stackTrace);
    } finally {
      session.reset();
    }
  }

  Future<void> _closeController() async {
    await _transcriptsController?.close();
    _transcriptsController = null;
  }
}
