import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_whisper_ggml/src/hallucination_filter.dart';
import 'package:flutter_whisper_ggml/src/whisper_stats.dart';

WhisperStats _stats({double noSpeech = 0.1, double logprob = -0.3}) =>
    WhisperStats(
      totalMs: 100,
      sampleMs: 0,
      encodeMs: 50,
      decodeMs: 5,
      batchdMs: 0,
      promptMs: 0,
      sampleCount: 32000,
      threadCount: 4,
      audioCtx: 256,
      tokenCount: 10,
      noSpeechProb: noSpeech,
      avgLogprob: logprob,
    );

void main() {
  const filter = HallucinationFilter();

  test('keeps a normal sentence', () {
    expect(
      filter.isHallucination('Bonjour, comment ça va ?', _stats()),
      isFalse,
    );
  });

  test('drops empty and punctuation-only text', () {
    expect(filter.isHallucination(''), isTrue);
    expect(filter.isHallucination(' ... !? '), isTrue);
  });

  test('drops known subtitle credits, whatever the case/punctuation', () {
    expect(
      filter.isHallucination(
        "Sous-titres réalisés par la communauté d'Amara.org",
      ),
      isTrue,
    );
    expect(filter.isHallucination("MERCI D'AVOIR REGARDÉ !"), isTrue);
  });

  test('keeps a real sentence that merely contains a blocklisted phrase', () {
    expect(
      filter.isHallucination(
        "Je lui ai dit merci d'avoir regardé mon dossier aussi rapidement hier",
      ),
      isFalse,
    );
  });

  test('applies the no-speech + low-confidence rule', () {
    expect(
      filter.isHallucination('Oui.', _stats(noSpeech: 0.9, logprob: -1.5)),
      isTrue,
    );
    // High no-speech alone is not enough when the decoder is confident.
    expect(
      filter.isHallucination('Oui.', _stats(noSpeech: 0.9, logprob: -0.2)),
      isFalse,
    );
  });

  test('drops repetition loops', () {
    expect(filter.isHallucination('je je je je je je je'), isTrue);
    expect(
      filter.isHallucination('je pense que je vais venir demain matin'),
      isFalse,
    );
  });

  test('HallucinationFilter.none only drops empty text', () {
    expect(HallucinationFilter.none.isHallucination('Abonnez-vous'), isFalse);
    expect(HallucinationFilter.none.isHallucination(''), isTrue);
  });
}
