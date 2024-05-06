from spacy.tokens import Doc

class NormalizedNounPhrases:
    def __init__(self):
        self._normalizedNounPhrases: dict = {}
        self._normalizedNounPhrasesFrequency: dict = {}
        self._totalFrequencies = None

    @property
    def normalizedNounPhrases(self):
        return self._normalizedNounPhrases

    @normalizedNounPhrases.setter
    def normalizedNounPhrasesSetter(self, value):
        self._normalizedNounPhrases = value

    @property
    def normalizedNounPhrasesFrequency(self):
        return self._normalizedNounPhrasesFrequency

    @normalizedNounPhrasesFrequency.setter
    def normalizedNounPhrasesFrequencySetter(self, value):
        self._normalizedNounPhrasesFrequency = value

    def add_normalized_noun_phrase(self, noun_phrase: Doc, normalized_noun_phrase: str):
        if normalized_noun_phrase in self._normalizedNounPhrasesFrequency:
            self._normalizedNounPhrasesFrequency[normalized_noun_phrase] += 1
        else:
            self._normalizedNounPhrases[noun_phrase] = normalized_noun_phrase
            self._normalizedNounPhrasesFrequency[normalized_noun_phrase] = 1

    def merge(self, other):
        for noun_phrase, normalized_noun_phrase in other.normalizedNounPhrases.items():
            self.add_normalized_noun_phrase(noun_phrase, normalized_noun_phrase)

    def get_frequency(self, normalized_noun_phrase: str) -> int:
        return self._normalizedNounPhrasesFrequency.get(normalized_noun_phrase)

    def get_normalized_frequency(self, normalized_noun_phrase: str):
        if self._totalFrequencies is None:
            self._totalFrequencies = max(self.normalizedNounPhrasesFrequency.values())
        return self.get_frequency(normalized_noun_phrase) / self._totalFrequencies

    def filter_by_normalized_frequency(self, threshold: float) -> 'NormalizedNounPhrases':
        filtered_noun_phrases = NormalizedNounPhrases()
        for noun_phrase, normalized_noun_phrase in self.normalizedNounPhrases.items():
            if self.get_normalized_frequency(normalized_noun_phrase) >= threshold:
                filtered_noun_phrases.add_normalized_noun_phrase(noun_phrase, normalized_noun_phrase)

        return filtered_noun_phrases
