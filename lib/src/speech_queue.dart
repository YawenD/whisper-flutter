import 'dart:typed_data';

/// Speech waiting for inference, with its position in the captured audio.
class QueuedSpeech {
  const QueuedSpeech(this.samples, this.start);

  final Float32List samples;

  /// Position of the first sample in the captured audio, in samples since
  /// listening started.
  final int start;

  /// Position just after the last sample.
  int get end => start + samples.length;
}

/// Bounded queue of speech segments, merged into batches for inference.
///
/// Pure Dart (no engine, no microphone) so the merging and dropping rules can
/// be tested on their own.
class SpeechQueue {
  SpeechQueue({
    required this.maxBatchSamples,
    required this.mergeGapSamples,
    required this.maxBacklogSamples,
  });

  /// A merged batch stays below this (the engine's window).
  final int maxBatchSamples;

  /// Silence inserted between merged segments so the engine sees the boundary.
  final int mergeGapSamples;

  /// Queued speech beyond this is dropped, oldest first.
  final int maxBacklogSamples;

  final List<QueuedSpeech> _items = [];
  int _samples = 0;

  bool get isEmpty => _items.isEmpty;
  bool get isNotEmpty => _items.isNotEmpty;

  void clear() {
    _items.clear();
    _samples = 0;
  }

  /// Adds [speech] and returns the number of samples dropped to stay within
  /// the backlog (0 most of the time). The newest segment is always kept.
  int add(QueuedSpeech speech) {
    if (speech.samples.isEmpty) return 0;
    _items.add(speech);
    _samples += speech.samples.length;
    var dropped = 0;
    while (_samples > maxBacklogSamples && _items.length > 1) {
      final oldest = _items.removeAt(0);
      _samples -= oldest.samples.length;
      dropped += oldest.samples.length;
    }
    return dropped;
  }

  /// Takes the oldest segments, merged up to [maxBatchSamples]. The batch
  /// spans from the first segment's start to the last one's end; its samples
  /// are the segments separated by [mergeGapSamples] of silence.
  QueuedSpeech take() {
    final parts = <QueuedSpeech>[_items.removeAt(0)];
    var total = parts.first.samples.length;
    while (_items.isNotEmpty &&
        total + mergeGapSamples + _items.first.samples.length <=
            maxBatchSamples) {
      total += mergeGapSamples + _items.first.samples.length;
      parts.add(_items.removeAt(0));
    }
    for (final p in parts) {
      _samples -= p.samples.length;
    }
    if (parts.length == 1) return parts.first;

    final merged = Float32List(total); // zero-filled gaps
    var offset = 0;
    for (final p in parts) {
      merged.setRange(offset, offset + p.samples.length, p.samples);
      offset += p.samples.length + mergeGapSamples;
    }
    return _Batch(merged, parts.first.start, parts.last.end);
  }
}

/// Merged segments: the gaps make its samples shorter than its span.
class _Batch extends QueuedSpeech {
  const _Batch(super.samples, super.start, this._end);

  final int _end;

  @override
  int get end => _end;
}
