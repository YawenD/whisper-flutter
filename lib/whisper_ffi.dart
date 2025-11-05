import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:wav/wav.dart';

typedef _WhisperInitNative = Pointer<Void> Function(Pointer<Utf8>);
typedef _WhisperInit = Pointer<Void> Function(Pointer<Utf8>);

typedef _WhisperFullNative =
    Int32 Function(
      Pointer<Void>, // ctx
      Pointer<Float>, // samples PCM float*
      Int32, // n_samples
    );
typedef _WhisperFull = int Function(Pointer<Void>, Pointer<Float>, int);

typedef _WhisperFreeNative = Void Function(Pointer<Void>);
typedef _WhisperFree = void Function(Pointer<Void>);

typedef _WhisperVersionNative = Pointer<Utf8> Function();
typedef _WhisperVersion = Pointer<Utf8> Function();

typedef _WhisperGetResultNative = Pointer<Utf8> Function(Pointer<Void>);
typedef _WhisperGetResult = Pointer<Utf8> Function(Pointer<Void>);

typedef _CStringFreeNative = Void Function(Pointer<Utf8>);
typedef _CStringFree = void Function(Pointer<Utf8>);

class WhisperFFI {
  late final DynamicLibrary _lib;
  late final _WhisperInit whisperInit;
  late final _WhisperFull whisperFull;
  late final _WhisperFree whisperFree;
  late final _WhisperVersion whisperVersion;
  late final _WhisperGetResult whisperGetResult;
  late final _CStringFree whisperFreeCstr;

  WhisperFFI() {
    _lib = Platform.isIOS
        ? DynamicLibrary.process()
        : DynamicLibrary.open('libwhisper.so');

    whisperInit = _lib.lookupFunction<_WhisperInitNative, _WhisperInit>(
      'whisper_init_bridge',
    );
    whisperFull = _lib.lookupFunction<_WhisperFullNative, _WhisperFull>(
      'whisper_full_bridge',
    );
    whisperFree = _lib.lookupFunction<_WhisperFreeNative, _WhisperFree>(
      'whisper_free_bridge',
    );
    whisperVersion = _lib
        .lookupFunction<_WhisperVersionNative, _WhisperVersion>(
          'whisper_version_bridge',
        );

    whisperGetResult = _lib
        .lookupFunction<_WhisperGetResultNative, _WhisperGetResult>(
          'whisper_get_result_bridge',
        );
    whisperFreeCstr = _lib.lookupFunction<_CStringFreeNative, _CStringFree>(
      'whisper_free_cstr_bridge',
    );
  }

  /// Facultatif : copie du modèle depuis les assets, etc.
  Future<String> prepareModel(String existingModelPath) async {
    // Si c'est déjà un chemin absolu vers un fichier existant, on le garde
    final maybeFile = File(existingModelPath);
    if (existingModelPath.startsWith('/') && await maybeFile.exists()) {
      return existingModelPath;
    }

    // Sinon, on suppose que c'est un asset Flutter → on le copie vers le FS
    final ByteData data = await rootBundle.load(existingModelPath);
    final Uint8List bytes = data.buffer.asUint8List();

    final Directory supportDir = await getApplicationSupportDirectory();
    final String outPath = '${supportDir.path}/model.bin';
    final outFile = File(outPath);

    // Évite les réécritures inutiles si la taille correspond
    if (!(await outFile.exists()) || (await outFile.length()) != bytes.length) {
      await outFile.create(recursive: true);
      await outFile.writeAsBytes(bytes, flush: true);
    }

    // Logs
    // ignore: avoid_print
    print('Model copied to: ' + outPath);
    return outPath;
  }

  /// Version simplifiée : l'utilisateur fournit le chemin du modèle et du fichier WAV.
  /// Le plugin décode le WAV → PCM → appelle Whisper.
  Future<String> transcribe({
    required String modelPath,
    required String wavPath,
  }) async {
    print(
      '[DART] Starting transcription with model: $modelPath, wav: $wavPath',
    );
    // 1️⃣ Charger le modèle
    final modelPtr = modelPath.toNativeUtf8();
    print('[DART] Calling whisperInit...');
    final ctx = whisperInit(modelPtr);
    print('[DART] whisperInit returned context: ${ctx.address}');
    malloc.free(modelPtr);

    // 2️⃣ Lire et décoder le fichier WAV
    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw Exception('Fichier audio introuvable : $wavPath');
    }
    final fileSize = await wavFile.length();
    print('[DART] WAV file size: $fileSize bytes');

    final bytes = await wavFile.readAsBytes();
    print('[DART] Bytes read: ${bytes.length}');
    print('[DART] First 32 bytes: ${bytes.take(32).toList()}');

    print('[DART] Decoding WAV file...');
    List<double> mono;

    try {
      // Try using the wav package first
      final wav = Wav.read(bytes);
      print('[DART] WAV decoded successfully with wav package');
      print('[DART] WAV number of channels: ${wav.channels.length}');

      for (var i = 0; i < wav.channels.length; i++) {
        print('[DART] Channel $i length: ${wav.channels[i].length}');
        if (wav.channels[i].length > 0) {
          print(
            '[DART] Channel $i first 5 samples: ${wav.channels[i].take(5).toList()}',
          );
          print(
            '[DART] Channel $i last 5 samples: ${wav.channels[i].skip(wav.channels[i].length - 5).take(5).toList()}',
          );
        }
      }

      // Convertir en mono
      if (wav.channels.isEmpty) {
        print('[DART] ERROR: No channels found in WAV file!');
        throw Exception('Fichier WAV invalide : aucun canal audio trouvé');
      } else if (wav.channels.length == 1) {
        print('[DART] Using mono channel directly');
        mono = wav.channels.first;
      } else {
        print('[DART] Converting ${wav.channels.length} channels to mono');
        mono = List<double>.generate(
          wav.channels.first.length,
          (i) => (wav.channels[0][i] + wav.channels[1][i]) / 2,
        );
      }

      print('[DART] Final mono samples count: ${mono.length}');

      // If the package decoded but got 0 samples, try manual decoding
      if (mono.isEmpty) {
        print(
          '[DART] Package decoded but got 0 samples, trying manual decode...',
        );
        mono = _decodeWavManually(bytes);
      }
    } catch (e) {
      print('[DART] Package wav failed: $e, trying manual decode...');
      // Fallback to manual decoding
      mono = _decodeWavManually(bytes);
    }

    // 3️⃣ Convertir la liste Dart -> buffer natif float*
    print('[DART] Converting ${mono.length} samples to native buffer...');
    final samplesPtr = malloc.allocate<Float>(sizeOf<Float>() * mono.length);
    for (var i = 0; i < mono.length; i++) {
      //       samplesPtr.elementAt(i).value = mono[i].toDouble();
      samplesPtr[i] = mono[i].toDouble();
    }
    print(
      '[DART] Sample buffer allocated: ${samplesPtr.address}, length: ${mono.length}',
    );

    // 4️⃣ Appeler whisper_full
    print('[DART] Calling whisperFull...');
    final fullResult = whisperFull(ctx, samplesPtr, mono.length);
    print('[DART] whisperFull returned: $fullResult');

    print('[DART] Calling whisperGetResult...');
    String result = '';
    final resPtr = whisperGetResult(ctx);
    print('[DART] whisperGetResult returned pointer: ${resPtr.address}');
    if (resPtr.address != 0) {
      result = resPtr.toDartString();
      // ignore: avoid_print
      print('[DART] Transcription result: $result');
      whisperFreeCstr(resPtr);
    } else {
      print('[DART] ERROR: whisperGetResult returned null pointer');
    }

    // 5️⃣ Libérer la mémoire
    malloc.free(samplesPtr);
    whisperFree(ctx);

    return result;
  }

  String version() => whisperVersion().toDartString();

  /// Decode WAV file manually to handle non-standard formats
  List<double> _decodeWavManually(Uint8List bytes) {
    print('[DART] Attempting manual WAV decoding...');

    if (bytes.length < 44) {
      throw Exception('WAV file too small');
    }

    // Check RIFF header
    final riff = String.fromCharCodes(bytes.sublist(0, 4));
    if (riff != 'RIFF') {
      throw Exception('Invalid WAV file: missing RIFF header');
    }

    // Check WAVE header
    final wave = String.fromCharCodes(bytes.sublist(8, 12));
    if (wave != 'WAVE') {
      throw Exception('Invalid WAV file: missing WAVE header');
    }

    int pos = 12;
    int? sampleRate;
    int? numChannels;
    int? bitsPerSample;
    int? dataStart;
    int? dataSize;

    // Parse chunks
    while (pos < bytes.length - 8) {
      final chunkId = String.fromCharCodes(bytes.sublist(pos, pos + 4));
      final chunkSize = bytes
          .sublist(pos + 4, pos + 8)
          .buffer
          .asByteData()
          .getUint32(0, Endian.little);

      print('[DART] Found chunk: $chunkId, size: $chunkSize at position $pos');

      if (chunkId == 'fmt ') {
        // Parse fmt chunk
        final audioFormat = bytes
            .sublist(pos + 8, pos + 10)
            .buffer
            .asByteData()
            .getUint16(0, Endian.little);
        numChannels = bytes
            .sublist(pos + 10, pos + 12)
            .buffer
            .asByteData()
            .getUint16(0, Endian.little);
        sampleRate = bytes
            .sublist(pos + 12, pos + 16)
            .buffer
            .asByteData()
            .getUint32(0, Endian.little);
        // byteRate and blockAlign are read but not used - needed to skip to bitsPerSample
        bytes
            .sublist(pos + 16, pos + 20)
            .buffer
            .asByteData()
            .getUint32(0, Endian.little); // byteRate
        bytes
            .sublist(pos + 20, pos + 22)
            .buffer
            .asByteData()
            .getUint16(0, Endian.little); // blockAlign
        bitsPerSample = bytes
            .sublist(pos + 22, pos + 24)
            .buffer
            .asByteData()
            .getUint16(0, Endian.little);

        print(
          '[DART] fmt chunk: format=$audioFormat, channels=$numChannels, sampleRate=$sampleRate, bitsPerSample=$bitsPerSample',
        );

        if (audioFormat != 1) {
          throw Exception(
            'Unsupported audio format: $audioFormat (only PCM supported)',
          );
        }
      } else if (chunkId == 'data') {
        // Found data chunk
        dataStart = pos + 8;
        dataSize = chunkSize;
        print('[DART] data chunk: start=$dataStart, size=$dataSize');
        break; // Found data chunk, we can stop
      }

      // Move to next chunk (chunk size + 8 bytes for chunk ID and size)
      pos += 8 + chunkSize;
      // Align to word boundary
      if (chunkSize % 2 == 1) pos++;
    }

    if (sampleRate == null || numChannels == null || bitsPerSample == null) {
      throw Exception('Missing fmt chunk in WAV file');
    }

    if (dataStart == null || dataSize == null) {
      throw Exception('Missing data chunk in WAV file');
    }

    // Decode PCM data
    final samples = <double>[];
    final bytesPerSample = bitsPerSample ~/ 8;

    // If dataSize is 0 or invalid, use the actual file size minus dataStart
    // This handles cases where the WAV header wasn't properly updated
    int actualDataSize = dataSize!;
    if (actualDataSize == 0 || actualDataSize > bytes.length - dataStart!) {
      actualDataSize = bytes.length - dataStart!;
      print(
        '[DART] WARNING: data chunk size ($dataSize) is invalid, using actual available data size: $actualDataSize',
      );
    }

    final totalSamples = actualDataSize ~/ bytesPerSample ~/ numChannels;

    print(
      '[DART] Decoding $totalSamples samples, $numChannels channels, $bitsPerSample bits per sample',
    );

    for (int i = 0; i < totalSamples; i++) {
      double sampleSum = 0.0;

      for (int ch = 0; ch < numChannels; ch++) {
        final sampleOffset =
            dataStart + (i * numChannels + ch) * bytesPerSample;

        if (sampleOffset + bytesPerSample > bytes.length) {
          break;
        }

        double sample;
        if (bitsPerSample == 16) {
          final sampleInt = bytes
              .sublist(sampleOffset, sampleOffset + 2)
              .buffer
              .asByteData()
              .getInt16(0, Endian.little);
          sample = sampleInt / 32768.0;
        } else if (bitsPerSample == 8) {
          final sampleInt = bytes[sampleOffset];
          sample = (sampleInt - 128) / 128.0;
        } else {
          throw Exception('Unsupported bits per sample: $bitsPerSample');
        }

        sampleSum += sample;
      }

      // Average channels for mono output
      samples.add(sampleSum / numChannels);
    }

    print('[DART] Manual decode successful: ${samples.length} samples');
    return samples;
  }
}
