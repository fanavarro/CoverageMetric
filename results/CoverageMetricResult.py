from pathlib import Path
from gensim.models.fasttext import save_facebook_model, FastText
from gensim.models import KeyedVectors, Word2Vec
from pandas import DataFrame


class Model:
    def __init__(self):
        self._name: str = None
        self._source_file = None
        self._embeddings_model:Word2Vec = None

    def __init__(self, model_name, source_file, embeddings_model:Word2Vec):
        self._name: str = model_name
        self._source_file = source_file
        self._embeddings_model = embeddings_model

    def __init__(self, source_file, embeddings_model:Word2Vec):
        self._name: str = Path(source_file).name
        self._source_file = source_file
        self._embeddings_model = embeddings_model

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def source_file(self):
        return self._source_file

    @source_file.setter
    def source_file(self, value: str):
        self._source_file = value

    @property
    def embeddings_model(self):
        return self._embeddings_model

    @embeddings_model.setter
    def model_name(self, value:Word2Vec):
        self._embeddings_model = value

    def save(self, file_path: Path):
        if isinstance(self.embeddings_model, FastText):
            fast_text_model: FastText = self.embeddings_model
            save_facebook_model(model=fast_text_model, path=str(file_path.absolute()))
