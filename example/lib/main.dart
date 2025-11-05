import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_whisper_ggml/whisper_ffi.dart';
import 'package:just_audio/just_audio.dart' as just_audio;
import 'package:path_provider/path_provider.dart';
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
  final just_audio.AudioPlayer _audioPlayer = just_audio.AudioPlayer();
  StreamSubscription<just_audio.PlayerState>? _playerSub;
  bool _isRecording = false;
  bool _isPlaying = false;
  String _text = 'Clique pour enregistrer';
  late WhisperFFI _whisper;
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

    // record n'a pas besoin d'open, on vérifie juste les permissions
    await _audioPlayer.setVolume(1.0);
    _playerSub = _audioPlayer.playerStateStream.listen((state) async {
      if (state.processingState == just_audio.ProcessingState.completed) {
        try {
          await _audioPlayer.stop();
        } catch (_) {}
        if (mounted) {
          setState(() {
            _isPlaying = false;
            _text = '⏹️ Lecture terminée';
          });
        }
      }
    });
    await Permission.microphone.request();
    await Permission.storage.request();
  }

  Future<void> _toggleRecording() async {
    final dir = await getApplicationDocumentsDirectory();
    final filePath = '${dir.path}/recorded.wav';

    if (!_isRecording) {
      // Démarrer l’enregistrement avec record (PCM16 WAV, 16kHz mono)
      final hasPerm = await _recorder.hasPermission();
      if (!hasPerm) {
        await Permission.microphone.request();
      }
      final canRecord = await _recorder.hasPermission();
      if (!canRecord) {
        setState(() => _text = 'Micro non autorisé');
        return;
      }
      await _recorder.start(
        RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
          bitRate: 16,
        ),
        path: filePath,
      );
      setState(() {
        _isRecording = true;
        _text = '🎙️ Enregistrement en cours...';
      });
    } else {
      // Stopper uniquement
      await _recorder.stop();
      setState(() {
        _isRecording = false;
        _text = '✅ Enregistrement terminé. Tu peux lire ou transcrire.';
      });
    }
  }

  Future<void> _togglePlayback() async {
    final dir = await getApplicationDocumentsDirectory();
    final filePath = '${dir.path}/recorded.wav';
    final exists = await File(filePath).exists();
    print('🔴 Path : $filePath');
    print('🔴 Exists : $exists');

    final file = File(filePath);
    print('WAV size: ${await file.length()} bytes');
    final bytes = await file.readAsBytes();
    print(bytes.take(32).toList());
    if (!exists) {
      setState(() => _text = 'Aucun enregistrement trouvé.');
      return;
    }
    if (!_isPlaying) {
      await _audioPlayer.setFilePath(filePath);
      await _audioPlayer.play();
      setState(() {
        _isPlaying = true;
        _text = '▶️ Lecture...';
      });
    } else {
      await _audioPlayer.stop();
      setState(() => _isPlaying = false);
    }
  }

  Future<void> _initWhisper() async {
    _whisper = WhisperFFI();
    _modelPath = await _whisper.prepareModel(
      'assets/models/ggml-tiny-q5_1.bin',
    );
    print('🧠 Modèle prêt: ' + _modelPath);
  }

  Future<void> _transcribeFile() async {
    final dir = await getApplicationDocumentsDirectory();
    final filePath = '${dir.path}/recorded.wav';
    final exists = await File(filePath).exists();
    if (!exists) {
      setState(() => _text = 'Aucun enregistrement à transcrire.');
      return;
    }
    // final convertedPath = await convertToPcm16(filePath);

    setState(() => _text = '⏳ Transcription en cours...');

    // Stopper la lecture si en cours (évite conflits session audio)
    if (_isPlaying) {
      try {
        await _audioPlayer.stop();
      } catch (_) {}
      if (mounted) setState(() => _isPlaying = false);
    }

    print('🧠 Transcription en cours...');
    final now = DateTime.now();
    final transcript = await _whisper.transcribe(
      modelPath: _modelPath,
      wavPath: filePath,
    );
    final duration = DateTime.now().difference(now).inMilliseconds;
    print('🧠 Temps d\'exécution : $duration ms');
    print('🧠 Transcription: ' + transcript);
    setState(() => _text = '🧠 Transcription :\n' + transcript);
  }

  @override
  void dispose() {
    _playerSub?.cancel();
    _audioPlayer.dispose();
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
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isRecording ? Colors.red : Colors.blue,
                  ),
                  onPressed: _toggleRecording,
                  child: Text(_isRecording ? 'Stopper' : 'Enregistrer'),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isPlaying ? Colors.orange : Colors.green,
                  ),
                  onPressed: _togglePlayback,
                  child: Text(_isPlaying ? 'Stop lecture' : 'Lire'),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _transcribeFile,
                  child: const Text('Transcrire'),
                ),
                const SizedBox(height: 20),
                Text(_text, textAlign: TextAlign.center),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
