from abc import ABC, abstractmethod
from pathlib import Path


class EmbeddingAlgorithm(ABC):

    @abstractmethod
    def generate_embedding_model(self, corpus_folder, vector_size=100, window=5, min_count=1, workers=4, negative=5,
                                 seed=1, iter=10):
        pass

    def save_embedding_model(self, model, output_folder, prefix_file_output):
        if output_folder is not None:
            output_path = Path(output_folder) / prefix_file_output
            model.save(str(output_path) + '.embeddings')
            model.wv.save_word2vec_format(str(output_path) + ".txt", binary=False)
            model.wv.save_word2vec_format(str(output_path) + ".bin", binary=True)
