import 'whisper_stats.dart';

/// Drops transcripts that Whisper produced without actual speech.
///
/// Small models are prone to "phantom" sentences on noise or silence, often
/// subtitle credits seen during training ("Sous-titres réalisés par la
/// communauté d'Amara.org", "Merci d'avoir regardé"...). Three checks:
/// 1. empty / punctuation-only text;
/// 2. OpenAI's rule: high no-speech probability AND low confidence;
/// 3. a blocklist of known phantom phrases (matched when they make up most of
///    the transcript, so a real sentence containing them survives).
/// Plus degenerate repetition loops ("je je je je je je").
class HallucinationFilter {
  const HallucinationFilter({
    this.noSpeechThreshold = 0.6,
    this.logprobThreshold = -1.0,
    this.blocklist = defaultBlocklist,
    this.maxRepeatRatio = 0.6,
  });

  /// Disables every check.
  static const HallucinationFilter none = HallucinationFilter(
    noSpeechThreshold: double.infinity,
    blocklist: [],
    maxRepeatRatio: double.infinity,
  );

  /// Drop when `noSpeechProb > noSpeechThreshold` and
  /// `avgLogprob < logprobThreshold` (same defaults as OpenAI's Whisper).
  final double noSpeechThreshold;
  final double logprobThreshold;

  /// Phrases (any case/punctuation) that are dropped when they make up the
  /// bulk of a transcript.
  final List<String> blocklist;

  /// Drop transcripts of 6+ words where a single word makes up more than this
  /// share of the words.
  final double maxRepeatRatio;

  static const List<String> defaultBlocklist = [
    "Sous-titres réalisés par la communauté d'Amara.org",
    'Sous-titres réalisés para la communauté d\'Amara.org',
    'Sous-titrage ST\' 501',
    'Sous-titrage Société Radio-Canada',
    'Sous-titrage FR',
    'Sous-titres par',
    "Merci d'avoir regardé cette vidéo",
    "Merci d'avoir regardé",
    'Abonnez-vous',
    "N'oubliez pas de vous abonner",
    'Thank you for watching',
    'Thanks for watching',
  ];

  /// Whether [text] should be discarded.
  bool isHallucination(String text, [WhisperStats? stats]) {
    final normalized = normalize(text);
    if (normalized.isEmpty) {
      return true;
    }

    if (stats != null &&
        stats.noSpeechProb > noSpeechThreshold &&
        stats.avgLogprob < logprobThreshold) {
      return true;
    }

    for (final phrase in blocklist) {
      final p = normalize(phrase);
      if (p.isNotEmpty &&
          normalized.contains(p) &&
          normalized.length <= p.length * 1.5) {
        return true;
      }
    }

    final words = normalized.split(' ');
    if (words.length >= 6) {
      final counts = <String, int>{};
      for (final w in words) {
        counts[w] = (counts[w] ?? 0) + 1;
      }
      final top = counts.values.reduce((a, b) => a > b ? a : b);
      if (top / words.length > maxRepeatRatio) {
        return true;
      }
    }
    return false;
  }

  /// Lowercase, letters/digits only, single spaces.
  static String normalize(String text) => text
      .toLowerCase()
      .replaceAll(RegExp(r'[^\p{L}\p{N}]+', unicode: true), ' ')
      .trim();
}
