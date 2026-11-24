import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_whisper_ggml/utils.dart';

void main() {
  test('converts little-endian PCM16 to floats', () {
    final bytes = Uint8List.fromList([0x00, 0x00, 0x00, 0x40, 0x00, 0x80]);
    expect(convertBytesToFloat32(bytes), [0.0, 0.5, -1.0]);
  });

  test('ignores a trailing odd byte instead of throwing', () {
    final bytes = Uint8List.fromList([0x00, 0x40, 0x12]);
    expect(convertBytesToFloat32(bytes), [0.5]);
  });

  test('honors the offset of a sub-view', () {
    final buffer = Uint8List.fromList([0xFF, 0xFF, 0x00, 0x40]);
    final view = Uint8List.sublistView(buffer, 2);
    expect(convertBytesToFloat32(view), [0.5]);
  });
}
