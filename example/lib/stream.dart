import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_whisper_ggml/flutter_whisper_ggml.dart';
import 'package:permission_handler/permission_handler.dart';

/// Engine override: --dart-define=LIVE_ENGINE=ggml|sherpa
const String _liveEngineOverride = String.fromEnvironment('LIVE_ENGINE');

/// ggml model override (whisper.cpp engine), e.g.
/// --dart-define=LIVE_MODEL=ggml-base-q4_0.bin
const String _liveModelOverride = String.fromEnvironment('LIVE_MODEL');

/// Partial results override: --dart-define=LIVE_PARTIALS=true|false
const String _livePartialsOverride = String.fromEnvironment('LIVE_PARTIALS');

/// Defaults measured with lib/bench.dart: iPhone runs whisper.cpp small on
/// Metal in ~0.4-0.75 s per sentence. Android is CPU-only: whisper.cpp base
/// takes ~2 s per sentence on a Pixel 9a, sherpa-onnx (ONNX Runtime) base fp32
/// ~0.4-1.5 s for a similar accuracy, and each partial result costs a full
/// inference.
final bool _useSherpa = _liveEngineOverride.isNotEmpty
    ? _liveEngineOverride == 'sherpa'
    : Platform.isAndroid;
final String _liveModel = _liveModelOverride.isNotEmpty
    ? _liveModelOverride
    : (Platform.isIOS ? 'ggml-small-q5_1.bin' : 'ggml-base-q5_1.bin');

WhisperModel get _whisperModel => _useSherpa
    ? const SherpaOnnxWhisperModel(
        encoder: 'assets/sherpa/whisper-base/base-encoder.onnx',
        decoder: 'assets/sherpa/whisper-base/base-decoder.onnx',
        tokens: 'assets/sherpa/whisper-base/base-tokens.txt',
      )
    : GgmlWhisperModel('assets/models/$_liveModel', useCoreMl: _liveCoreMl);

String get _modelLabel =>
    _useSherpa ? 'sherpa-onnx base fp32' : 'whisper.cpp $_liveModel';
final bool _livePartials = _livePartialsOverride.isNotEmpty
    ? _livePartialsOverride == 'true'
    : Platform.isIOS;

/// Core ML encoder on iOS when bundled (--dart-define=LIVE_COREML=true).
/// Off by default: Metal measured as accurate and fast enough on recent
/// iPhones.
const bool _liveCoreMl = bool.fromEnvironment('LIVE_COREML');

void main() => runApp(const WhisperExampleApp());

class WhisperExampleApp extends StatefulWidget {
  const WhisperExampleApp({super.key});

  @override
  State<WhisperExampleApp> createState() => _WhisperExampleAppState();
}

class _WhisperExampleAppState extends State<WhisperExampleApp>
    with WidgetsBindingObserver {
  final WhisperLiveTranscriber _transcriber = WhisperLiveTranscriber(
    vadAdapter: VadAdapter(settings: const VadSettings()),
    decodingOptions: const WhisperDecodingOptions(
      // Domain vocabulary helps small models a lot; adapt to the app.
      initialPrompt: 'Conversation en français.',
    ),
    enablePartialResults: _livePartials,
  );
  final List<StreamSubscription<String>> _subscriptions = [];
  final List<String> _transcripts = [];
  String _partial = '';
  bool _isReady = false;
  bool _isListening = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Both streams live as long as the transcriber: subscribe once.
    _subscriptions
      ..add(
        _transcriber.transcripts.listen(
          (text) => setState(() {
            _transcripts.add(text);
            _partial = '';
          }),
          onError: (Object error) =>
              setState(() => _error = 'Erreur transcription : $error'),
        ),
      )
      ..add(
        _transcriber.partialTranscripts.listen(
          (text) => setState(() => _partial = text),
        ),
      );
    WidgetsBinding.instance.addPostFrameCallback((_) => _init());
  }

  Future<void> _init() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      setState(() {
        _error = 'Micro non autorisé. Activez la permission pour continuer.';
      });
      return;
    }

    try {
      await _transcriber.prepare(
        model: _whisperModel,
        modelPath: 'assets/models/silero_vad.onnx',
      );
      // Loads and warms up the model now, so "Démarrer" is instant.
      await _transcriber.initSession();
      setState(() => _isReady = true);
    } catch (e) {
      setState(() => _error = 'Initialisation impossible : $e');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Without background audio the OS cuts the microphone anyway: stop
    // cleanly so the speech in progress is still transcribed.
    if (state == AppLifecycleState.paused && _isListening) {
      _stopListening();
    }
  }

  Future<void> _startListening() async {
    if (!_isReady || _isListening) return;
    try {
      await _transcriber.startListening();
      setState(() {
        _isListening = true;
        _error = null;
      });
    } catch (e) {
      setState(() => _error = 'Impossible de démarrer l\'écoute : $e');
    }
  }

  Future<void> _stopListening() async {
    if (!_isListening) return;
    setState(() => _isListening = false);
    try {
      // Returns once the last sentence has been transcribed.
      await _transcriber.stopListening();
    } catch (e) {
      setState(() => _error = 'Erreur lors de l\'arrêt : $e');
    }
    setState(() => _partial = '');
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    for (final s in _subscriptions) {
      s.cancel();
    }
    _transcriber.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text(
            '$_modelLabel${_livePartials ? '' : ' · sans partiels'}',
          ),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (!_isReady && _error == null)
                const Text(
                  'Chargement du modèle…',
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
                child: Text(_isListening ? 'Écoute en cours…' : 'Démarrer'),
              ),
              ElevatedButton(
                onPressed: _isListening ? _stopListening : null,
                child: const Text('Arrêter'),
              ),
              const SizedBox(height: 24),
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.blueGrey),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: _transcripts.isEmpty && _partial.isEmpty
                      ? const Center(
                          child: Text(
                            'Les transcriptions apparaîtront ici.',
                            textAlign: TextAlign.center,
                          ),
                        )
                      : ListView(
                          padding: const EdgeInsets.all(12),
                          children: [
                            for (final text in _transcripts)
                              Padding(
                                padding: const EdgeInsets.only(bottom: 8),
                                child: Text('• $text'),
                              ),
                            if (_partial.isNotEmpty)
                              Text(
                                '… $_partial',
                                style: const TextStyle(color: Colors.grey),
                              ),
                          ],
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
