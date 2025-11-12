import 'dart:typed_data';

import 'package:record/record.dart';

abstract class IRecordAdapter {
  Future<Stream<Uint8List>> startStream();
  Future<void> stopStream();
  Future<void> dispose();
}

class RecordAdapter implements IRecordAdapter {
  final AudioRecorder _audioRecorder = AudioRecorder();

  @override
  Future<Stream<Uint8List>> startStream() async {
    return await _audioRecorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        bitRate: 16,
      ),
    );
  }

  @override
  Future<void> stopStream() async {
    await _audioRecorder.stop();
  }

  @override
  Future<void> dispose() async {
    await _audioRecorder.dispose();
  }
}
