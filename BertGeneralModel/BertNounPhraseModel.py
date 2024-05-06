# https://colab.research.google.com/drive/1neTWJoDkhO2mCKRgqG1klVIkU5Qk3fQh?usp=sharing
# https://github.com/sf-wa-326/phrase-bert-topic-model
import threading


import time
from pathlib import Path
import numpy
from sentence_transformers import SentenceTransformer
from scipy import spatial

EMBEDDING_DIM = 768
MODEL_PATH = Path(__file__).parent / '..' / 'resources' / 'phrase-bert-model' / 'pooled_context_para_triples_p=0.8'

class BertNounPhraseModel(object):
    _instance = None
    _lock = threading.Lock()
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(BertNounPhraseModel, cls).__new__(cls)
        return cls._instance

    def aa(cls):
        if cls._instance is None:
            cls._instance = super(BertNounPhraseModel, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH.absolute().as_posix(), device='cpu')

    def get_embeddings(self, phrases: list):
        with self._lock:
            embeddings = self._cache.get(phrases)
            if embeddings is None:
                embeddings = self.model.encode(phrases)
                self._cache[phrases] = embeddings

        return embeddings
    def get_embeddings2(self, phrases: list):
        return self.model.encode(phrases)

    @staticmethod
    def get_mean_point(embeddings: list):
        return numpy.mean(embeddings, axis=0)

    def get_cosine_similarity(self, a, b):
        return 1 - spatial.distance.cosine(a, b)

    def get_similarity(self, phrases1: tuple, phrases2: tuple):
        """
        Get the similarity between the set of phrases phrases1 and phrases2 by getting the mean point in the bert
        general model and by computing the cosine similarity between both points
        @param phrases1: First set of phrases
        @param phrases2: Second set of Phrases
        @return: The cosine similarity between the mean point from phrases1 and the mean point from phrases2.
        """
        #print(f'{phrases1}\n{phrases2}\n\n')
        embeddings1 = self.get_embeddings(phrases1)
        mean_point1 = BertNounPhraseModel.get_mean_point(embeddings1)

        embeddings2 = self.get_embeddings(phrases2)
        mean_point2 = BertNounPhraseModel.get_mean_point(embeddings2)

        return self.get_cosine_similarity(mean_point1, mean_point2)
