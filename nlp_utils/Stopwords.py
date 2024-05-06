from pathlib import Path

from gensim import utils


def read_stopwords(file):
    with open(file) as f:
        return [word.strip().lower() for word in f.read().splitlines()]


def get_stop_words():
    return STOPWORDS


def stopwords_rule(word: str, count: int, min_count: int):
    if word.lower() in get_stop_words():
        return utils.RULE_DISCARD
    return utils.RULE_DEFAULT


STOPWORDS_FILE = Path('resources') / 'nlp-resources' / 'stopwords.txt'
STOPWORDS = read_stopwords(STOPWORDS_FILE)
