import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_whisper_ggml/src/speech_queue.dart';

QueuedSpeech speech(int start, int length, [double value = 1]) =>
    QueuedSpeech(Float32List(length)..fillRange(0, length, value), start);

void main() {
  SpeechQueue queue({int batch = 100, int gap = 10, int backlog = 1000}) =>
      SpeechQueue(
        maxBatchSamples: batch,
        mergeGapSamples: gap,
        maxBacklogSamples: backlog,
      );

  test('a single segment keeps its own span', () {
    final q = queue()..add(speech(500, 40));
    final batch = q.take();
    expect([batch.start, batch.end, batch.samples.length], [500, 540, 40]);
    expect(q.isEmpty, isTrue);
  });

  test('merged segments span from the first start to the last end', () {
    final q = queue()
      ..add(speech(100, 30, 1))
      ..add(speech(200, 20, 2))
      ..add(speech(400, 30, 3));
    final batch = q.take();
    expect(batch.start, 100);
    expect(batch.end, 430);
    // 30 + gap 10 + 20 + gap 10 + 30 samples, the gaps silent.
    expect(batch.samples.length, 100);
    expect(batch.samples[29], 1);
    expect(batch.samples[30], 0);
    expect(batch.samples[40], 2);
    expect(batch.samples[70], 3);
  });

  test('a batch stops before the engine window', () {
    final q = queue(batch: 60)
      ..add(speech(0, 30))
      ..add(speech(100, 30))
      ..add(speech(200, 10));
    final first = q.take();
    expect([first.start, first.end], [0, 30]);
    final second = q.take();
    expect([second.start, second.end], [100, 210]);
  });

  test('the backlog drops the oldest speech and reports it', () {
    final q = queue(backlog: 50);
    expect(q.add(speech(0, 30)), 0);
    expect(q.add(speech(100, 30)), 30);
    expect(q.take().start, 100);
    expect(q.isEmpty, isTrue);
  });

  test('the newest segment is kept even beyond the backlog', () {
    final q = queue(backlog: 10);
    expect(q.add(speech(0, 40)), 0);
    expect(q.take().end, 40);
  });

  test('empty speech is ignored', () {
    final q = queue()..add(speech(0, 0));
    expect(q.isEmpty, isTrue);
  });
}
