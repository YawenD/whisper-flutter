import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_whisper_ggml/record_adapter.dart';
import 'package:flutter_whisper_ggml/src/whisper_session_worker.dart';
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

  WhisperSessionWorker? _sessionWorker;
  StreamSubscription<Uint8List>? _recordSubscription;
  StreamController<String>? _transcriptsController;
  Future<void> _processing = Future.value();
  bool _listening = false;
  bool _vadReady = false;
  String? _modelPath;
  bool _captureEnabled = true;

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
    if (_sessionWorker != null) {
      return;
    }
    final modelPath = _modelPath;
    if (modelPath == null) {
      throw StateError(
        'prepare() must be called with a model path before initSession().',
      );
    }
    _sessionWorker = await WhisperSessionWorker.spawn(modelPath);
  }

  Future<void> startListening() async {
    if (_listening) {
      throw StateError('Live transcription already started.');
    }
    _captureEnabled = true;
    final modelPath = _modelPath;
    if (modelPath == null) {
      throw StateError(
        'prepare() doit être appelé avec un chemin de modèle avant startListening().',
      );
    }

    _transcriptsController = StreamController<String>.broadcast();
    if (_sessionWorker == null) {
      _sessionWorker = await WhisperSessionWorker.spawn(modelPath);
    } else {
      await _sessionWorker!.reset();
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
    _captureEnabled = false;

    await _recordSubscription?.cancel();
    _recordSubscription = null;
    await _recordAdapter.stopStream();

    await _processing;

    if (!_vadReady) {
      await _flushPendingTranscript();
    }

    await _closeController();
    // Keep the worker (and native context) alive to avoid reloading
    // the model on the next startListening(). It will be released
    // in dispose().
    await _sessionWorker?.reset();
  }

  Future<void> dispose() async {
    await stopListening();
    await _recordAdapter.dispose();
    await _vadAdapter.dispose();
  }

  Future<void> _handleChunk(Uint8List chunk) async {
    final session = _sessionWorker;
    if (session == null) {
      return;
    }
    if (!_captureEnabled) {
      return;
    }

    if (!_vadReady) {
      await session.appendPcmBytes(Uint8List.fromList(chunk));
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

        final transcript = (await session.transcribeFloatSamples(
          speechSamples,
        )).trim();

        if (transcript.isNotEmpty) {
          _transcriptsController?.add(transcript);
        }
      }
    } catch (error, stackTrace) {
      _transcriptsController?.addError(error, stackTrace);
    }
  }

  Future<void> _flushPendingTranscript() async {
    final session = _sessionWorker;
    if (session == null) {
      return;
    }

    try {
      final transcript = (await session.transcribeBuffered()).trim();
      if (transcript.isNotEmpty) {
        _transcriptsController?.add(transcript);
      }
    } catch (error, stackTrace) {
      _transcriptsController?.addError(error, stackTrace);
    } finally {
      await session.reset();
    }
  }

  Future<void> pauseCapture() async {
    if (!_listening) {
      return;
    }
    _captureEnabled = false;
    await _processing;
    await _sessionWorker?.reset();
  }

  Future<void> resumeCapture() async {
    if (!_listening) {
      return;
    }
    _captureEnabled = true;
  }

  Future<void> _closeController() async {
    await _transcriptsController?.close();
    _transcriptsController = null;
  }
}
