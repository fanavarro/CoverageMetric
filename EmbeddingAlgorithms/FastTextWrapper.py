from gensim import utils
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.word2vec import PathLineSentences
from pathlib import Path

from EmbeddingAlgorithms.EmbeddingAlgorithm import EmbeddingAlgorithm
from nlp_utils import Stopwords


class FastTextWrapper(EmbeddingAlgorithm):
    OUTPUT_FILE_NAME = 'word2vec'

    def __init__(self):
        pass

    def generate_embedding_model(self, corpus_folder, vector_size=100, window=5, min_count=1, workers=4, negative=5,
                                 seed=1, iter=10):
        sentences = PathLineSentences(corpus_folder)
        model = FastText(sentences=sentences, vector_size=vector_size, window=window,
                         min_count=min_count, workers=workers, negative=negative, seed=seed, epochs=iter)
                         #trim_rule=Stopwords.stopwords_rule
        return model
