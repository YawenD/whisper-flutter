// Copyright (c)  2024  Xiaomi Corporation
import "dart:io";
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

// Copy the asset file from src to dst
Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  dst ??= basename(src);
  final target = join(directory.path, dst);

  // Check existence BEFORE loading the asset into RAM.
  if (await File(target).exists()) {
    return target;
  }

  final data = await rootBundle.load(src);
  final List<int> bytes = data.buffer.asUint8List(
    data.offsetInBytes,
    data.lengthInBytes,
  );
  await File(target).writeAsBytes(bytes);

  return target;
}

Float32List convertBytesToFloat32(Uint8List bytes, [endian = Endian.little]) {
  final values = Float32List(bytes.length ~/ 2);

  final data = ByteData.view(bytes.buffer);

  for (var i = 0; i < bytes.length; i += 2) {
    int short = data.getInt16(i, endian);
    values[i ~/ 2] = short / 32768.0;
  }

  return values;
}
