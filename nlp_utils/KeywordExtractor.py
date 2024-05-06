import pathlib

import spacy
from spacy.tokens import Doc

NLP = spacy.load("en_core_web_sm")
sep_tokens = ['-', '_', '/']


def remove_stop_words(chunk: Doc) -> list:
    without_stop_words_tokens = []
    for token in chunk:
        if not token.is_stop and (token.text in sep_tokens or not token.is_punct):
            without_stop_words_tokens.append(token)

    return without_stop_words_tokens


def normalize_noun_phrase(chunk: Doc) -> str:
    token_list = remove_stop_words(chunk)
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
    return cleaned_chunk.strip()


def get_keywords(raw_text):
    doc = NLP(raw_text)
    for x in doc.ents:
        y = normalize_noun_phrase(x)

