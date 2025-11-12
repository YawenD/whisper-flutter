import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_whisper_ggml/flutter_whisper_ggml.dart';
import 'package:permission_handler/permission_handler.dart';

void main() => runApp(const WhisperExampleApp());

class WhisperExampleApp extends StatefulWidget {
  const WhisperExampleApp({super.key});

  @override
  State<WhisperExampleApp> createState() => _WhisperExampleAppState();
}

class _WhisperExampleAppState extends State<WhisperExampleApp> {
  late final WhisperLiveTranscriber _transcriber;
  StreamSubscription<String>? _transcriptSubscription;
  final List<String> _transcripts = [];
  bool _isReady = false;
  bool _isListening = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _init();
    });
  }

  Future<void> _init() async {
    _transcriber = WhisperLiveTranscriber();

    final hasPerm = await Permission.microphone.isGranted;
    if (!hasPerm) {
      await Permission.microphone.request().then((status) {
        if (!status.isGranted) {
          setState(() {
            _error =
                'Micro non autorisé. Activez la permission pour continuer.';
          });
          return;
        }
      });
    }

    await _transcriber.prepare(
      modelAssetPath: 'assets/models/ggml-tiny-q5_1.bin',
      modelPath: 'assets/models/silero_vad.onnx',
    );

    setState(() {
      _isReady = true;
    });
  }

  Future<void> _startListening() async {
    if (!_isReady || _isListening) return;

    try {
      await _transcriber.startListening();
      _transcriptSubscription = _transcriber.transcripts.listen(
        (text) {
          if (text.trim().isEmpty) return;
          setState(() {
            _transcripts.add(text);
          });
        },
        onError: (Object error) {
          setState(() {
            _error = 'Erreur transcription: $error';
          });
        },
      );
      setState(() {
        _isListening = true;
        _error = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Impossible de démarrer l\'écoute: $e';
      });
    }
  }

  Future<void> _stopListening() async {
    if (!_isListening) return;

    await _transcriptSubscription?.cancel();
    _transcriptSubscription = null;

    try {
      await _transcriber.stopListening();
    } catch (e) {
      setState(() {
        _error = 'Erreur lors de l\'arrêt: $e';
      });
    }

    setState(() {
      _isListening = false;
    });
  }

  @override
  void dispose() {
    _transcriptSubscription?.cancel();
    _transcriber.dispose();
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
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (!_isReady)
                  const Text(
                    'Initialisation de Whisper en cours...',
                    textAlign: TextAlign.center,
                  ),
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Text(
                      _error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ElevatedButton(
                  onPressed: _isReady && !_isListening ? _startListening : null,
                  child: Text(_isListening ? 'Écoute en cours...' : 'Démarrer'),
                ),
                ElevatedButton(
                  onPressed: _stopListening,
                  child: const Text('Arrêter'),
                ),
                const SizedBox(height: 24),
                Expanded(
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.blueGrey),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: _transcripts.isEmpty
                        ? const Center(
                            child: Text(
                              'Les transcriptions apparaîtront ici.',
                              textAlign: TextAlign.center,
                            ),
                          )
                        : ListView.separated(
                            padding: const EdgeInsets.all(12),
                            itemBuilder: (context, index) {
                              final text = _transcripts[index];
                              return Text('• $text');
                            },
                            separatorBuilder: (_, __) =>
                                const SizedBox(height: 8),
                            itemCount: _transcripts.length,
                          ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
