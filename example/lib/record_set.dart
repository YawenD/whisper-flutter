// Records a labelled test set with a real voice, for lib/bench.dart.
//
//   flutter run --release -t lib/record_set.dart
//
// Read each sentence shown on screen, tap "Stop", go on. Every take is saved
// as a 16 kHz mono PCM16 WAV next to a references.tsv file in:
//   Android: /sdcard/Android/data/<package>/files/bench_real/  (adb pull)
//   iOS:     Documents/bench_real/  (xcrun devicectl device copy from ...)
// Copy that folder to example/assets/bench_real/ (git-ignored: it is a voice
// recording) and run the bench with --dart-define=BENCH_SET=real.

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

/// Typical assistant requests: questions, commands, names, numbers, short
/// and long sentences.
const List<String> _sentences = [
  "Bonjour, comment tu vas aujourd'hui ?",
  "Rappelle-moi d'appeler ma mère demain à dix-huit heures.",
  'Quel temps va-t-il faire ce week-end à Lyon ?',
  'Ajoute du lait, des œufs et du beurre à ma liste de courses.',
  "Est-ce que tu peux m'expliquer comment fonctionne une batterie ?",
  'Envoie un message à Julien pour lui dire que je serai en retard.',
  'Mets un minuteur de vingt minutes pour les pâtes.',
  'Je voudrais réserver une table pour quatre personnes samedi soir.',
  'Combien font trois cent quarante-cinq divisé par cinq ?',
  'Raconte-moi une histoire courte pour endormir un enfant.',
  "Non, ce n'est pas ce que je voulais dire, recommence.",
  "Quelle est la capitale de l'Australie ?",
  "J'ai rendez-vous chez le dentiste mardi prochain à neuf heures trente, "
      'tu peux le noter ?',
  'Traduis bonne nuit en espagnol et en italien.',
  "Merci beaucoup, c'est parfait, à plus tard !",
];

void main() => runApp(const MaterialApp(home: RecordSetPage()));

class RecordSetPage extends StatefulWidget {
  const RecordSetPage({super.key});

  @override
  State<RecordSetPage> createState() => _RecordSetPageState();
}

class _RecordSetPageState extends State<RecordSetPage> {
  final AudioRecorder _recorder = AudioRecorder();
  final BytesBuilder _audio = BytesBuilder(copy: false);
  StreamSubscription<Uint8List>? _subscription;
  Directory? _dir;
  int _index = 0;
  bool _recording = false;
  String _status = '';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await Permission.microphone.request();
    final base = Platform.isAndroid
        ? (await getExternalStorageDirectory())!
        : await getApplicationDocumentsDirectory();
    final dir = Directory('${base.path}/bench_real')
      ..createSync(recursive: true);
    // Resume after the last saved take.
    var next = 0;
    while (File('${dir.path}/${_name(next)}').existsSync()) {
      next++;
    }
    setState(() {
      _dir = dir;
      _index = next;
      _status = 'Dossier : ${dir.path}';
    });
  }

  static String _name(int i) => '${(i + 1).toString().padLeft(2, '0')}.wav';

  Future<void> _toggle() async {
    if (_recording) {
      await _subscription?.cancel();
      await _recorder.stop();
      _save();
      setState(() => _recording = false);
      return;
    }
    _audio.clear();
    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ),
    );
    _subscription = stream.listen(_audio.add);
    setState(() => _recording = true);
  }

  void _save() {
    final dir = _dir!;
    final pcm = _audio.takeBytes();
    File('${dir.path}/${_name(_index)}').writeAsBytesSync(_wav(pcm));
    _writeReferences(dir);
    setState(() {
      _status =
          '${_name(_index)} enregistré '
          '(${(pcm.length / 32000).toStringAsFixed(1)} s)';
      _index++;
    });
  }

  void _writeReferences(Directory dir) {
    final lines = <String>[
      for (var i = 0; i < _sentences.length; i++)
        if (File('${dir.path}/${_name(i)}').existsSync())
          '${_name(i)}\t${_sentences[i]}',
    ];
    File(
      '${dir.path}/references.tsv',
    ).writeAsStringSync('${lines.join('\n')}\n');
  }

  void _redoPrevious() {
    if (_index == 0 || _recording) return;
    setState(() => _index--);
  }

  /// 16 kHz mono PCM16 WAV.
  static Uint8List _wav(Uint8List pcm) {
    final header = ByteData(44);
    void ascii(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        header.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    ascii(0, 'RIFF');
    header.setUint32(4, 36 + pcm.length, Endian.little);
    ascii(8, 'WAVE');
    ascii(12, 'fmt ');
    header.setUint32(16, 16, Endian.little);
    header.setUint16(20, 1, Endian.little); // PCM
    header.setUint16(22, 1, Endian.little); // mono
    header.setUint32(24, 16000, Endian.little);
    header.setUint32(28, 32000, Endian.little); // byte rate
    header.setUint16(32, 2, Endian.little); // block align
    header.setUint16(34, 16, Endian.little); // bits per sample
    ascii(36, 'data');
    header.setUint32(40, pcm.length, Endian.little);
    return (BytesBuilder()
          ..add(header.buffer.asUint8List())
          ..add(pcm))
        .takeBytes();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final done = _index >= _sentences.length;
    return Scaffold(
      appBar: AppBar(title: const Text('Jeu de test vocal')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              done
                  ? 'Terminé : ${_sentences.length} phrases enregistrées.'
                  : 'Phrase ${_index + 1} / ${_sentences.length}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 24),
            if (!done)
              Text(
                _sentences[_index],
                style: Theme.of(context).textTheme.headlineSmall,
              ),
            const SizedBox(height: 32),
            if (!done)
              FilledButton(
                onPressed: _dir == null ? null : _toggle,
                style: FilledButton.styleFrom(
                  backgroundColor: _recording ? Colors.red : null,
                  minimumSize: const Size.fromHeight(64),
                ),
                child: Text(
                  _recording ? 'Stop' : 'Enregistrer',
                  style: const TextStyle(fontSize: 20),
                ),
              ),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: _index == 0 || _recording ? null : _redoPrevious,
              child: const Text('Refaire la phrase précédente'),
            ),
            const SizedBox(height: 24),
            Text(
              'Parle normalement, comme à l\'assistant, à ta distance '
              'habituelle du téléphone.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            Text(_status, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    );
  }
}
