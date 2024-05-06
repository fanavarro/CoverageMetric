import spacy
from spacy.tokens import Doc
from pathlib import Path
import os
from results.NormalizedNounPhrases import NormalizedNounPhrases

NLP = spacy.load("en_core_web_sm")
SPACY_MAX_CHARS_LIMIT = 1000000
sep_tokens = ['-', '_', '/']


def remove_stop_words(chunk: Doc) -> list:
    without_stop_words_tokens = []
    for token in chunk:
        if not token.is_stop and (token.text in sep_tokens or not token.is_punct):
            without_stop_words_tokens.append(token)

    return without_stop_words_tokens


def remove_special_chars(text: str) -> str:
    cleaned_text = text.replace("'", "")
    cleaned_text = cleaned_text.replace('"', "")
    cleaned_text = cleaned_text.replace('\\', '')
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text


def normalize_noun_phrase(chunk: Doc) -> str:
    chunk_labels = [entity.label_ for entity in chunk.ents]
    #print(f'{chunk.text} -> {",".join(chunk_labels)}')
    # Skip noun phrases if they are people. warning: galactose, cdc11... are labelled as person :S
    if 'PERSON' in chunk_labels or 'et al' in chunk.text:
        return ''

    token_list = remove_stop_words(chunk)
    # Skip noun phrases if they are numbers.
    if len(token_list) == 1 and token_list[0].is_digit:
        return ''

    cleaned_chunk = ''
    for i in range(0, len(token_list)):
        if token_list[i].text == '\n':
            cleaned_chunk = cleaned_chunk.strip() + ' '
        elif token_list[i].text in sep_tokens:
            cleaned_chunk = cleaned_chunk.strip() + token_list[i].text
        elif token_list[i].tag_ in {"NNS", "NNPS"}:
            cleaned_chunk = cleaned_chunk + token_list[i].lemma_ + ' '
        else:
            cleaned_chunk = cleaned_chunk + token_list[i].text + ' '

    # Remove special characters (owlready2 fails to load annotations with these characters.)
    cleaned_chunk = remove_special_chars(cleaned_chunk)
    return cleaned_chunk.strip()


def get_normalized_noun_phrases(raw_text, only_keywords=False, to_lower_case=False) -> NormalizedNounPhrases:
    # Set maximum text length for Spacy
    NLP.max_length = max(SPACY_MAX_CHARS_LIMIT, len(raw_text) + 250)
    doc = NLP(raw_text)
    normalized_noun_phrases: NormalizedNounPhrases = NormalizedNounPhrases()
    if only_keywords:
        normalized_keywords = get_normalized_keywords(doc)

    for chunk in doc.noun_chunks:
        #print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
        normalized_noun_phrase = normalize_noun_phrase(chunk)
        if to_lower_case:
            normalized_noun_phrase = normalized_noun_phrase.lower()

        # Skip noun phrases that are a stop word
        if normalized_noun_phrase.strip() != '':
            # print(f'{chunk.text}\t->\t{chunk.root.text}, {chunk.root.pos_}\t->\t {normalized_noun_phrase}')
            # normalized_noun_phrases[chunk] = normalized_noun_phrase
            if not only_keywords:
                normalized_noun_phrases.add_normalized_noun_phrase(chunk, normalized_noun_phrase)

            elif normalized_noun_phrase in normalized_keywords:
                normalized_noun_phrases.add_normalized_noun_phrase(chunk, normalized_noun_phrase)

    return normalized_noun_phrases


def get_normalized_keywords(doc) -> NormalizedNounPhrases:
    normalized_keywords = set()
    for keyword in doc.ents:
        normalized_keyword = normalize_noun_phrase(keyword)
        if normalized_keyword.strip() != '':
            normalized_keywords.add(normalized_keyword)
    return normalized_keywords


def normalize_text(raw_text, normalized_noun_phrases: NormalizedNounPhrases) -> str:
    normalized_text = raw_text
    normalized_noun_phrases_dict = normalized_noun_phrases.normalizedNounPhrases
    reverse_keys = list(normalized_noun_phrases_dict.keys())[::-1]
    for noun_phrase in reverse_keys:
        normalized_noun_phrase = normalized_noun_phrases_dict[noun_phrase]
        start = noun_phrase.start_char
        end = noun_phrase.end_char
        normalized_text = normalized_text[:start] + normalized_noun_phrase + normalized_text[end:]
    return remove_special_chars(normalized_text)


def normalize_text_to_file(raw_text, normalized_noun_phrases: NormalizedNounPhrases, output_file_path: Path):
    if not output_file_path.parent.exists():
        os.makedirs(output_file_path.parent)
    with open(str(output_file_path), 'w+') as output_file:
        normalized_text = normalize_text(raw_text, normalized_noun_phrases)
        output_file.write(normalized_text)