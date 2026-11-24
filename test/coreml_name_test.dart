import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_whisper_ggml/whisper_method_channel.dart';

void main() {
  test('Core ML encoder name mirrors whisper.cpp', () {
    expect(
      WhisperMethodChannel.coreMlEncoderName('ggml-base-q5_1.bin'),
      'ggml-base-encoder.mlmodelc',
    );
    expect(
      WhisperMethodChannel.coreMlEncoderName('assets/models/ggml-small.bin'),
      'ggml-small-encoder.mlmodelc',
    );
    expect(
      WhisperMethodChannel.coreMlEncoderName('ggml-large-v3-turbo-q8_0.bin'),
      'ggml-large-v3-turbo-encoder.mlmodelc',
    );
  });
}
