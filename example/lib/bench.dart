// Benchmark screen for the native Whisper pipeline.
//
// ALWAYS run it in release mode, otherwise the numbers are meaningless:
//   flutter run --release -t lib/bench.dart
// Auto-run on startup (results are also printed as "[BENCH] ..." log lines):
//   flutter run --release -t lib/bench.dart --dart-define=AUTO_BENCH=true
//
// Every ggml-*.bin found under assets/models/ can be benchmarked: drop a model
// file there and it shows up in the dropdown. The "A/B" switch runs every clip
// twice, with the dynamic encoder context and with the full 30 s window, to
// check the speed gain does not cost accuracy (compare the WER). Reference clips live in
// assets/bench/ and their expected transcripts in [_references].

import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_whisper_ggml/flutter_whisper_ggml.dart';
import 'package:flutter_whisper_ggml/utils.dart' show copyAssetFile;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:wav/wav.dart';

const bool _autoBench = bool.fromEnvironment('AUTO_BENCH');

/// Model picked in auto mode, e.g. --dart-define=BENCH_MODEL=ggml-base-q5_1.bin
const String _benchModel = String.fromEnvironment('BENCH_MODEL');

/// Several runs in one launch (auto mode), "model:metal|coreml[:threads]"
/// separated by commas, e.g.
/// --dart-define=BENCH_PLAN=ggml-small-q5_1.bin:metal,ggml-small-q5_1.bin:coreml
/// --dart-define=BENCH_PLAN=ggml-base-q5_1.bin:cpu:2,ggml-base-q5_1.bin:cpu:6
const String _benchPlan = String.fromEnvironment('BENCH_PLAN');

/// Runs per clip in auto mode (default 3).
const int _benchRuns = int.fromEnvironment('BENCH_RUNS', defaultValue: 3);

/// iOS: run the encoder with Core ML when a `<model>-encoder.mlmodelc` is bundled
/// (--dart-define=BENCH_COREML=false to force Metal for comparison).
const bool _benchCoreMl = bool.fromEnvironment(
  'BENCH_COREML',
  defaultValue: true,
);

/// Expected transcripts of the reference clips, used to compute the WER.
const Map<String, String> _references = {
  'assets/bench/01_court.wav': 'Bonjour, comment ça va ?',
  'assets/bench/02_moyen.wav':
      'Je voudrais réserver une table pour quatre personnes demain soir vers '
      'vingt heures.',
  'assets/bench/03_long.wav':
      'Hier, nous sommes allés au marché pour acheter des légumes frais. '
      'Ensuite, nous avons préparé une soupe et nous avons invité nos voisins '
      'à dîner avec nous, ce qui était très agréable.',
  'assets/bench/04_rdv.wav':
      "Est-ce que tu peux m'appeler vers dix-sept heures trente ?",
  'assets/bench/05_meteo.wav':
      'Il va pleuvoir toute la journée, prends ton parapluie.',
  'assets/bench/06_courses.wav':
      "Il faut acheter du pain, trois tomates, du fromage et une bouteille "
      "d'eau.",
  'assets/bench/07_travail.wav':
      'La réunion de lundi est reportée à mercredi matin, merci de prévenir '
      "toute l'équipe avant ce soir.",
};

void main() => runApp(const MaterialApp(home: BenchPage()));

class BenchPage extends StatefulWidget {
  const BenchPage({super.key});

  @override
  State<BenchPage> createState() => _BenchPageState();
}

class _ClipResult {
  _ClipResult({
    required this.config,
    required this.clip,
    required this.runs,
    required this.text,
    required this.wer,
  });

  final String config;
  final String clip;
  final List<WhisperStats> runs;
  final String text;
  final double wer;

  double _median(double Function(WhisperStats) f) {
    final values = runs.map(f).toList()..sort();
    return values[values.length ~/ 2];
  }

  double get audioMs => runs.first.audioMs;
  double get totalMs => _median((t) => t.totalMs);
  double get maxTotalMs => runs.map((t) => t.totalMs).reduce(math.max);
  double get encodeMs => _median((t) => t.encodeMs);
  double get decodeMs => _median((t) => t.decodeMs);
  double get rtf => totalMs / audioMs;
  WhisperStats get last => runs.last;
}

class _BenchPageState extends State<BenchPage> {
  List<String> _models = [];
  List<String> _clips = [];
  String? _model;
  int _runs = 3;
  bool _compareCtx = false;
  bool _coreMl = _benchCoreMl;
  int _threads = 0;

  /// When set, [_run] goes through this model's engine instead of [_model].
  WhisperModel? _engineModel;
  final TextEditingController _prompt = TextEditingController();
  bool _running = false;
  String _status = 'Chargement des assets…';
  String? _systemInfo;
  double? _loadMs;
  double? _coldMs;
  final List<_ClipResult> _results = [];

  @override
  void initState() {
    super.initState();
    _loadAssets();
  }

  @override
  void dispose() {
    _prompt.dispose();
    super.dispose();
  }

  Future<void> _loadAssets() async {
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final assets = manifest.listAssets();
    setState(() {
      _models =
          assets
              .where(
                (a) =>
                    a.startsWith('assets/models/ggml-') && a.endsWith('.bin'),
              )
              .toList()
            ..sort();
      _clips =
          assets
              .where((a) => a.startsWith('assets/bench/') && a.endsWith('.wav'))
              .toList()
            ..sort();
      _model = _models.isEmpty
          ? null
          : _models.firstWhere(
              (m) => _benchModel.isNotEmpty && m.endsWith('/$_benchModel'),
              orElse: () => _models.firstWhere(
                (m) => m.contains('tiny'),
                orElse: () => _models.first,
              ),
            );
      _status = _models.isEmpty
          ? 'Aucun modèle ggml-*.bin dans assets/models/'
          : 'Prêt (${_clips.length} clips de référence)';
    });
    if (kReleaseMode == false) {
      setState(() {
        _status =
            '⚠️ Mode ${kProfileMode ? 'profile' : 'debug'} : mesures non '
            'représentatives, lance avec --release.';
      });
    }
    if (_autoBench && _benchPlan.isNotEmpty) {
      for (final step in _benchPlan.split(',')) {
        final parts = step.trim().split(':');
        if (parts.length >= 2 && parts[1] == 'worker') {
          // "whisper-base:worker" → the app's path: WhisperSessionWorker +
          // sherpa-onnx engine, fp32 files of assets/sherpa/whisper-base/.
          final size = parts.first.replaceFirst('whisper-', '');
          final dir = 'assets/sherpa/${parts.first}';
          setState(() {
            _engineModel = SherpaOnnxWhisperModel(
              encoder: '$dir/$size-encoder.onnx',
              decoder: '$dir/$size-decoder.onnx',
              tokens: '$dir/$size-tokens.txt',
            );
            _model = 'sherpa-worker:${parts.first}';
            _runs = _benchRuns;
          });
          await _run();
          setState(() => _engineModel = null);
          continue;
        }
        if (parts.length >= 2 && parts[1] == 'sherpa') {
          // "whisper-base:sherpa:4:int8|fp32|mixed" → assets/sherpa/whisper-base/
          final threads = parts.length >= 3 ? int.tryParse(parts[2]) ?? 4 : 4;
          final precision = parts.length >= 4 ? parts[3] : 'int8';
          setState(() => _runs = _benchRuns);
          await _runSherpa(parts.first, threads, precision);
          continue;
        }
        final model = _models.where((m) => m.endsWith('/${parts.first}'));
        if (model.isEmpty) {
          _log('error=unknown model ${parts.first}');
          continue;
        }
        setState(() {
          _model = model.first;
          _coreMl = parts.length >= 2 && parts[1] == 'coreml';
          _threads = parts.length >= 3 ? int.tryParse(parts[2]) ?? 0 : 0;
          _runs = _benchRuns;
        });
        await _run();
      }
      _log('all_done');
    } else if (_autoBench && _model != null) {
      await _run();
      _log('all_done');
    }
  }

  /// Same clips and metrics, but transcribed by sherpa-onnx (ONNX Runtime,
  /// int8 Whisper export) instead of whisper.cpp, on the UI isolate (the UI
  /// freezes during the run; fine for a benchmark). The model folder
  /// `assets/sherpa/<name>/` holds `<size>-encoder.int8.onnx`,
  /// `<size>-decoder.int8.onnx` and `<size>-tokens.txt`.
  /// [precision]: `int8` (both int8), `fp32` (both fp32) or `mixed` (fp32
  /// encoder, int8 decoder).
  Future<void> _runSherpa(String name, int threads, String precision) async {
    final label = 'sherpa-$precision';
    setState(() {
      _running = true;
      _results.clear();
    });
    sherpa.OfflineRecognizer? recognizer;
    try {
      _setStatus('sherpa-onnx : copie du modèle $name…');
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      final prefix = 'assets/sherpa/$name/';
      final files = <String, String>{};
      for (final asset in manifest.listAssets().where(
        (a) => a.startsWith(prefix),
      )) {
        final file = asset.substring(prefix.length);
        files[file] = await copyAssetFile(asset, 'sherpa/$name/$file');
      }
      String pick(String suffix) =>
          files.entries.firstWhere((e) => e.key.endsWith(suffix)).value;
      final encoderSuffix = precision == 'int8'
          ? '-encoder.int8.onnx'
          : '-encoder.onnx';
      final decoderSuffix = precision == 'fp32'
          ? '-decoder.onnx'
          : '-decoder.int8.onnx';

      _setStatus('sherpa-onnx : chargement…');
      final loadWatch = Stopwatch()..start();
      sherpa.initBindings();
      recognizer = sherpa.OfflineRecognizer(
        sherpa.OfflineRecognizerConfig(
          model: sherpa.OfflineModelConfig(
            whisper: sherpa.OfflineWhisperModelConfig(
              encoder: pick(encoderSuffix),
              decoder: pick(decoderSuffix),
              language: 'fr',
              task: 'transcribe',
            ),
            tokens: pick('-tokens.txt'),
            numThreads: threads,
            debug: false,
          ),
        ),
      );
      loadWatch.stop();
      _log(
        'model=sherpa:$name precision=$precision threads=$threads '
        'load=${loadWatch.elapsedMilliseconds}ms',
      );

      (String, double) decode(Float32List samples) {
        final stream = recognizer!.createStream();
        try {
          stream.acceptWaveform(samples: samples, sampleRate: 16000);
          final watch = Stopwatch()..start();
          recognizer.decode(stream);
          watch.stop();
          return (
            recognizer.getResult(stream).text.trim(),
            watch.elapsedMicroseconds / 1000,
          );
        } finally {
          stream.free();
        }
      }

      final clips = <String, Float32List>{
        for (final c in _clips) c: await _loadClip(c),
      };
      final cold = decode(clips.values.first);
      _log('cold_first_run total=${cold.$2.toStringAsFixed(0)}ms');

      for (final entry in clips.entries) {
        final runs = <WhisperStats>[];
        var text = '';
        for (var i = 0; i < _runs; i++) {
          _setStatus('sherpa · ${_short(entry.key)} : run ${i + 1}/$_runs');
          // Let the UI repaint the status between synchronous decodes.
          await Future<void>.delayed(const Duration(milliseconds: 16));
          final (t, ms) = decode(entry.value);
          text = t;
          runs.add(
            WhisperStats(
              totalMs: ms,
              sampleMs: 0,
              encodeMs: 0,
              decodeMs: 0,
              batchdMs: 0,
              promptMs: 0,
              sampleCount: entry.value.length,
              threadCount: threads,
              audioCtx: 1500,
              tokenCount: 0,
              noSpeechProb: 0,
              avgLogprob: 0,
            ),
          );
        }
        final result = _ClipResult(
          config: label,
          clip: entry.key,
          runs: runs,
          text: text,
          wer: _wer(_references[entry.key], text),
        );
        setState(() => _results.add(result));
        _log(
          'config="$label" clip=${_short(entry.key)} '
          'audio=${result.audioMs.toStringAsFixed(0)}ms '
          'median=${result.totalMs.toStringAsFixed(0)}ms '
          'max=${result.maxTotalMs.toStringAsFixed(0)}ms '
          'wer=${(result.wer * 100).toStringAsFixed(0)}% text="$text"',
        );
      }
      final wer =
          _results.map((r) => r.wer).reduce((a, b) => a + b) / _results.length;
      final total = _results.map((r) => r.totalMs).reduce((a, b) => a + b);
      final worst = _results.map((r) => r.maxTotalMs).reduce(math.max);
      _log(
        'summary config="$label" mean_wer=${(wer * 100).toStringAsFixed(1)}% '
        'sum_median=${total.toStringAsFixed(0)}ms '
        'worst=${worst.toStringAsFixed(0)}ms',
      );
      _setStatus('Terminé (sherpa)');
    } catch (e) {
      _setStatus('Erreur sherpa : $e');
      _log('error=$e');
    } finally {
      recognizer?.free();
      if (mounted) setState(() => _running = false);
    }
  }

  Future<Float32List> _loadClip(String asset) async {
    final data = await rootBundle.load(asset);
    final wav = Wav.read(data.buffer.asUint8List());
    if (wav.samplesPerSecond != 16000 || wav.channels.length != 1) {
      throw StateError('$asset must be 16 kHz mono');
    }
    return Float32List.fromList(wav.channels.first);
  }

  Future<void> _run() async {
    final model = _model;
    if (model == null || _running) return;
    setState(() {
      _running = true;
      _results.clear();
      _systemInfo = null;
      _loadMs = null;
      _coldMs = null;
    });

    final prompt = _prompt.text.trim().isEmpty ? null : _prompt.text.trim();
    final configs = <String, WhisperDecodingOptions>{
      '30 s': WhisperDecodingOptions(
        initialPrompt: prompt,
        threads: _threads,
      ),
      if (_compareCtx) ...{
        'dyn m128': WhisperDecodingOptions(
          dynamicAudioCtx: true,
          initialPrompt: prompt,
        ),
        'dyn m256': WhisperDecodingOptions(
          dynamicAudioCtx: true,
          initialPrompt: prompt,
          audioCtxMargin: 256,
        ),
        'dyn m128+fb': WhisperDecodingOptions(
          dynamicAudioCtx: true,
          initialPrompt: prompt,
          temperatureInc: 0.4,
        ),
        'dyn m256+fb': WhisperDecodingOptions(
          dynamicAudioCtx: true,
          initialPrompt: prompt,
          audioCtxMargin: 256,
          temperatureInc: 0.4,
        ),
        '30 s+fb': WhisperDecodingOptions(
          dynamicAudioCtx: false,
          initialPrompt: prompt,
          temperatureInc: 0.4,
        ),
      },
    };

    WhisperSessionWorker? worker;
    try {
      _setStatus('Copie du modèle…');
      final engineModel = _engineModel;
      final spec = engineModel != null
          ? await engineModel.resolve()
          : WhisperEngineSpec.ggml(
              await WhisperMethodChannel().prepareModel(
                model,
                useCoreMl: _coreMl,
              ),
            );

      _setStatus('Chargement du modèle…');
      final loadWatch = Stopwatch()..start();
      worker = await WhisperSessionWorker.spawnEngine(
        spec,
        language: 'fr',
        options: configs.values.first,
      );
      final info = await worker.systemInfo();
      loadWatch.stop();
      setState(() {
        _systemInfo = info;
        _loadMs = loadWatch.elapsedMicroseconds / 1000;
      });
      _log(
        'model=$model coreml=${_coreMl ? 'on' : 'off'} '
        'threads=${_threads == 0 ? 'auto' : _threads} '
        'load=${_loadMs!.toStringAsFixed(0)}ms',
      );
      _log('system_info=$info');

      final clips = <String, Float32List>{
        for (final c in _clips) c: await _loadClip(c),
      };

      // First inference pays one-off allocations (and Metal pipeline builds on
      // iOS): report it separately and keep it out of the medians.
      _setStatus('Warm-up…');
      final cold = await worker.warmUp();
      setState(() => _coldMs = cold?.totalMs);
      _log('cold_first_run $cold');

      for (final config in configs.entries) {
        await worker.configure(config.value);
        for (final entry in clips.entries) {
          final runs = <WhisperStats>[];
          var text = '';
          for (var i = 0; i < _runs; i++) {
            _setStatus(
              '${config.key} · ${_short(entry.key)} : run ${i + 1}/$_runs',
            );
            final r = await worker.transcribe(entry.value);
            if (r.stats != null) runs.add(r.stats!);
            text = r.text;
          }
          if (runs.isEmpty) continue;
          final result = _ClipResult(
            config: config.key,
            clip: entry.key,
            runs: runs,
            text: text,
            wer: _wer(_references[entry.key], text),
          );
          setState(() => _results.add(result));
          _log(
            'config="${config.key}" clip=${_short(entry.key)} '
            'audio=${result.audioMs.toStringAsFixed(0)}ms '
            'median=${result.totalMs.toStringAsFixed(0)}ms '
            'max=${result.maxTotalMs.toStringAsFixed(0)}ms '
            'rtf=${result.rtf.toStringAsFixed(3)} '
            'encode=${result.encodeMs.toStringAsFixed(0)}ms '
            'decode=${result.decodeMs.toStringAsFixed(1)}ms/tok '
            'ctx=${result.last.audioCtx} '
            'nosp=${result.last.noSpeechProb.toStringAsFixed(2)} '
            'logp=${result.last.avgLogprob.toStringAsFixed(2)} '
            'threads=${result.last.threadCount} '
            'wer=${(result.wer * 100).toStringAsFixed(0)}% '
            'text="${result.text}"',
          );
        }
      }
      for (final config in configs.keys) {
        final rs = _results.where((r) => r.config == config).toList();
        if (rs.isEmpty) continue;
        final wer = rs.map((r) => r.wer).reduce((a, b) => a + b) / rs.length;
        final total = rs.map((r) => r.totalMs).reduce((a, b) => a + b);
        final worst = rs.map((r) => r.maxTotalMs).reduce(math.max);
        _log(
          'summary config="$config" mean_wer=${(wer * 100).toStringAsFixed(1)}% '
          'sum_median=${total.toStringAsFixed(0)}ms '
          'worst=${worst.toStringAsFixed(0)}ms',
        );
      }
      _setStatus('Terminé');
      _log('done');
    } catch (e) {
      _setStatus('Erreur : $e');
      _log('error=$e');
    } finally {
      await worker?.dispose();
      if (mounted) setState(() => _running = false);
    }
  }

  void _setStatus(String s) => setState(() => _status = s);

  // print() on purpose: these lines must also show up in release logs.
  // ignore: avoid_print
  void _log(String line) => print('[BENCH] $line');

  static String _short(String asset) => asset.split('/').last;

  /// Word error rate after lowercasing and stripping punctuation.
  static double _wer(String? reference, String hypothesis) {
    if (reference == null) return double.nan;
    List<String> words(String s) => s
        .toLowerCase()
        .replaceAll(RegExp(r"[^\p{L}\p{N}' ]", unicode: true), ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
    final r = words(reference);
    final h = words(hypothesis);
    if (r.isEmpty) return h.isEmpty ? 0 : 1;
    var prev = List<int>.generate(h.length + 1, (j) => j);
    for (var i = 1; i <= r.length; i++) {
      final cur = List<int>.filled(h.length + 1, 0)..[0] = i;
      for (var j = 1; j <= h.length; j++) {
        final cost = r[i - 1] == h[j - 1] ? 0 : 1;
        cur[j] = math.min(
          math.min(cur[j - 1] + 1, prev[j] + 1),
          prev[j - 1] + cost,
        );
      }
      prev = cur;
    }
    return prev[h.length] / r.length;
  }

  @override
  Widget build(BuildContext context) {
    final mono = Theme.of(
      context,
    ).textTheme.bodySmall?.copyWith(fontFamily: 'monospace');
    return Scaffold(
      appBar: AppBar(title: const Text('Whisper bench')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Row(
            children: [
              Expanded(
                child: DropdownButton<String>(
                  isExpanded: true,
                  value: _model,
                  items: [
                    for (final m in _models)
                      DropdownMenuItem(value: m, child: Text(_short(m))),
                  ],
                  onChanged: _running
                      ? null
                      : (m) => setState(() => _model = m),
                ),
              ),
              const SizedBox(width: 12),
              DropdownButton<int>(
                value: _runs,
                items: [
                  for (final n in const [1, 3, 5, 10])
                    DropdownMenuItem(value: n, child: Text('$n runs')),
                ],
                onChanged: _running
                    ? null
                    : (n) => setState(() => _runs = n ?? _runs),
              ),
            ],
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Encodeur Core ML (iOS, si fourni)'),
            value: _coreMl,
            onChanged: _running ? null : (v) => setState(() => _coreMl = v),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Comparer les variantes (ctx, marge, repli)'),
            value: _compareCtx,
            onChanged: _running
                ? null
                : (v) => setState(() => _compareCtx = v),
          ),
          TextField(
            controller: _prompt,
            enabled: !_running,
            decoration: const InputDecoration(
              labelText: 'Prompt initial (optionnel)',
              hintText: 'Vocabulaire du domaine, noms propres…',
            ),
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _running || _model == null ? null : _run,
            child: Text(_running ? 'En cours…' : 'Lancer le bench'),
          ),
          const SizedBox(height: 12),
          Text(_status),
          if (_loadMs != null)
            Text('Chargement modèle : ${_loadMs!.toStringAsFixed(0)} ms'),
          if (_coldMs != null)
            Text('1re inférence (froid) : ${_coldMs!.toStringAsFixed(0)} ms'),
          if (_systemInfo != null) ...[
            const SizedBox(height: 8),
            Text(_systemInfo!, style: mono),
          ],
          const SizedBox(height: 12),
          for (final r in _results)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${r.config} · ${_short(r.clip)} — audio '
                      '${(r.audioMs / 1000).toStringAsFixed(1)} s',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                    Text(
                      'médiane ${r.totalMs.toStringAsFixed(0)} ms '
                      '(max ${r.maxTotalMs.toStringAsFixed(0)}) · '
                      'RTF ${r.rtf.toStringAsFixed(3)}\n'
                      'encode ${r.encodeMs.toStringAsFixed(0)} ms · '
                      'decode ${r.decodeMs.toStringAsFixed(1)} ms/token · '
                      '${r.last.threadCount} threads · ctx ${r.last.audioCtx}\n'
                      'WER ${(r.wer * 100).toStringAsFixed(0)} % · '
                      'no-speech ${r.last.noSpeechProb.toStringAsFixed(2)} · '
                      'logprob ${r.last.avgLogprob.toStringAsFixed(2)}',
                      style: mono,
                    ),
                    const SizedBox(height: 4),
                    Text('« ${r.text} »'),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
