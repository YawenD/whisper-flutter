import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_whisper_ggml/src/whisper_session_worker.dart';

void main() {
  test('glues the hyphens NeMo tokenizers space out', () {
    expect(
      normalizeNemoText('Rappelle - moi à dix - huit heures.'),
      'Rappelle-moi à dix-huit heures.',
    );
    expect(
      normalizeNemoText('Quel temps va - t - il faire ce week - end ?'),
      'Quel temps va-t-il faire ce week-end ?',
    );
  });

  test('leaves regular text and dashes at the edges alone', () {
    expect(normalizeNemoText('Bonjour, ça va ?'), 'Bonjour, ça va ?');
    expect(normalizeNemoText('- Bonjour'), '- Bonjour');
  });
}
