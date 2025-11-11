import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_whisper_ggml/flutter_whisper_ggml.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

void main() => runApp(const WhisperExampleApp());

class WhisperExampleApp extends StatefulWidget {
  const WhisperExampleApp({super.key});

  @override
  State<WhisperExampleApp> createState() => _WhisperExampleAppState();
}

class _WhisperExampleAppState extends State<WhisperExampleApp> {
  final AudioRecorder _recorder = AudioRecorder();
  late Whisper _whisper;
  late final String _modelPath;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _init();
    });
  }

  Future<void> _init() async {
    await _initWhisper();
    await Permission.microphone.request();
  }

  Future<void> _startSpeechReconization() async {
    await _recorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
        bitRate: 16,
      ),
    );
  }

  Future<void> _stopSpeechReconization() async {
    await _recorder.stop();
  }

  Future<void> _initWhisper() async {
    _whisper = Whisper();
    _modelPath = await _whisper.prepareModel(
      'assets/models/ggml-tiny-q5_1.bin',
    );
    print('🧠 Modèle prêt: ' + _modelPath);
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Whisper Microphone Demo')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _startSpeechReconization,
                  child: const Text('Démarrer la stream'),
                ),
                ElevatedButton(
                  onPressed: _stopSpeechReconization,
                  child: const Text('Stopper la stream'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
