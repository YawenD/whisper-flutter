import 'dart:async';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter_whisper_ggml/src/ios_whisper_ffi.dart';

class WhisperSessionWorker {
  WhisperSessionWorker._(this._sendPort, this._isolate);

  final SendPort _sendPort;
  final Isolate _isolate;
  bool _isDisposed = false;

  static Future<WhisperSessionWorker> spawn(String modelPath) async {
    final readyPort = ReceivePort();
    final isolate = await Isolate.spawn(
      _WhisperSessionIsolate.entryPoint,
      <dynamic>[modelPath, readyPort.sendPort],
      debugName: 'WhisperSessionWorker',
    );

    // The isolate sends the SendPort only AFTER loading the model,
    // so spawn() blocks until the model is ready.
    final result = await readyPort.first;
    readyPort.close();

    if (result is Map && result['error'] != null) {
      isolate.kill(priority: Isolate.immediate);
      throw Exception(result['error'] as String);
    }

    return WhisperSessionWorker._(result as SendPort, isolate);
  }

  Future<void> appendPcmBytes(Uint8List bytes) async {
    await _send({'type': _WhisperSessionMessage.appendPcm, 'bytes': bytes});
  }

  Future<String> transcribeFloatSamples(List<double> samples) async {
    final result = await _send({
      'type': _WhisperSessionMessage.transcribeFloat,
      'samples': samples,
    });
    return (result as String?) ?? '';
  }

  Future<String> transcribeBuffered() async {
    final result = await _send({
      'type': _WhisperSessionMessage.transcribeBuffered,
    });
    return (result as String?) ?? '';
  }

  Future<void> reset() async {
    await _send({'type': _WhisperSessionMessage.reset});
  }

  Future<void> dispose() async {
    if (_isDisposed) {
      return;
    }
    await _send({'type': _WhisperSessionMessage.dispose});
    _isolate.kill(priority: Isolate.immediate);
    _isDisposed = true;
  }

  Future<dynamic> _send(Map<String, dynamic> message) async {
    if (_isDisposed) {
      throw StateError('Worker already disposed');
    }
    final responsePort = ReceivePort();
    _sendPort.send(<String, dynamic>{
      ...message,
      'reply': responsePort.sendPort,
    });
    final dynamic result = await responsePort.first;
    responsePort.close();

    if (result is Map && result['error'] != null) {
      throw Exception(result['error'] as String);
    }

    return result;
  }
}

class _WhisperSessionMessage {
  static const String appendPcm = 'appendPcm';
  static const String transcribeFloat = 'transcribeFloat';
  static const String transcribeBuffered = 'transcribeBuffered';
  static const String reset = 'reset';
  static const String dispose = 'dispose';
}

class _WhisperSessionIsolate {
  static void entryPoint(List<dynamic> args) async {
    final modelPath = args[0] as String;
    final SendPort handshakePort = args[1] as SendPort;
    final commandPort = ReceivePort();

    // Load the model BEFORE the handshake so that spawn() blocks
    // until the model is ready.
    final WhisperStreamSession session;
    try {
      session = WhisperStreamSession.create(modelPath);
    } catch (e) {
      handshakePort.send(<String, dynamic>{'error': e.toString()});
      return;
    }

    handshakePort.send(commandPort.sendPort);

    await for (final dynamic raw in commandPort) {
      if (raw is! Map) {
        continue;
      }
      final type = raw['type'] as String?;
      final SendPort? replyPort = raw['reply'] as SendPort?;

      Future<void> respond(dynamic value) async {
        replyPort?.send(value);
      }

      try {
        switch (type) {
          case _WhisperSessionMessage.appendPcm:
            final bytes = raw['bytes'] as Uint8List? ?? Uint8List(0);
            session.appendPcmBytes(bytes);
            await respond(null);
            break;
          case _WhisperSessionMessage.transcribeFloat:
            final samples = (raw['samples'] as List<dynamic>? ?? const [])
                .cast<double>();
            if (samples.isEmpty) {
              await respond('');
              break;
            }
            session.appendFloatSamples(samples);
            final transcript = session.transcribeSync().trim();
            session.reset();
            await respond(transcript);
            break;
          case _WhisperSessionMessage.transcribeBuffered:
            final transcript = session.transcribeSync().trim();
            session.reset();
            await respond(transcript);
            break;
          case _WhisperSessionMessage.reset:
            session.reset();
            await respond(null);
            break;
          case _WhisperSessionMessage.dispose:
            session.dispose();
            await respond(null);
            commandPort.close();
            return;
          default:
            await respond(<String, dynamic>{
              'error': 'Unknown worker message: $type',
            });
        }
      } on Object catch (error, stackTrace) {
        if (kDebugMode) {
          // Comments stay in English per repo convention.
          print('[WhisperSessionWorker] Error: $error\n$stackTrace');
        }
        await respond(<String, dynamic>{'error': error.toString()});
      }
    }
  }
}
