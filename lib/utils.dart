// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

/// Copy the asset [src] to the app support directory and return its path.
///
/// The copy is atomic (written to a temporary file, then renamed), so an app
/// killed mid-copy never leaves a truncated file that would be reused forever.
/// A sidecar `.complete` file records the copied size: when it matches, the
/// asset is not even loaded again (models can weigh hundreds of MB).
///
/// Assets are identified by file name: give a new name to an updated model
/// (e.g. `ggml-base-q5_1-v2.bin`) so devices pick up the new file.
Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  final target = File(join(directory.path, dst ?? basename(src)));
  final marker = File('${target.path}.complete');

  if (await target.exists() && await marker.exists()) {
    final expected = int.tryParse((await marker.readAsString()).trim());
    if (expected != null && expected == await target.length()) {
      return target.path;
    }
  }

  final data = await rootBundle.load(src);
  final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  await target.parent.create(recursive: true);
  final tmp = File('${target.path}.tmp');
  await tmp.writeAsBytes(bytes, flush: true);
  await tmp.rename(target.path);
  await marker.writeAsString('${bytes.length}', flush: true);
  return target.path;
}

/// Convert PCM16 little-endian bytes to float samples in [-1, 1).
/// A trailing odd byte (incomplete sample) is ignored.
Float32List convertBytesToFloat32(Uint8List bytes, [endian = Endian.little]) {
  final sampleCount = bytes.length ~/ 2;
  final values = Float32List(sampleCount);

  // sublistView honors the list's offset into its underlying buffer, unlike
  // ByteData.view(bytes.buffer) which would read from offset 0.
  final data = ByteData.sublistView(bytes);

  for (var i = 0; i < sampleCount; i++) {
    values[i] = data.getInt16(i * 2, endian) / 32768.0;
  }

  return values;
}
