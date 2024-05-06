import tempfile
from pathlib import Path

from owl2vec_star import owl2vec_star

from EmbeddingAlgorithms.EmbeddingAlgorithm import EmbeddingAlgorithm


class Owl2VecStarWrapper(EmbeddingAlgorithm):
    OUTPUT_FILE_NAME = 'owl2vec'
    CONFIG_TEMPLATE_PATH = Path(__file__).parent / '..' / 'resources' / 'owl2vec-star-resources' / 'default.cfg'

    def __init__(self):
        self.cache_dir = tempfile.TemporaryDirectory(prefix='owl2vec_cache')

    def __del__(self):
        self.cache_dir.cleanup()

    def generate_embedding_model(self, ontology_file, vector_size=100, window=5, min_count=1, workers=4, negative=5,
                                 seed=1, iter=10, output_folder=None):
        model = None
        config_tmp_name = None
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as config_tmp,\
                open(Owl2VecStarWrapper.CONFIG_TEMPLATE_PATH) as config_template:
            config_content = "".join(config_template.readlines())
            config_content = config_content.replace('${ontology_file}', str(ontology_file))
            config_content = config_content.replace('${vector_size}', str(vector_size))
            config_content = config_content.replace('${iter}', str(iter))
            config_content = config_content.replace('${window}', str(window))
            config_content = config_content.replace('${min_count}', str(min_count))
            config_content = config_content.replace('${negative}', str(negative))
            config_content = config_content.replace('${seed}', str(seed))
            config_content = config_content.replace('${cache_dir}', str(self.cache_dir.name))
            # config_content = config_content.replace('${cache_dir}', 'owl2vec_cache')
            config_tmp.write(config_content)
            config_tmp_name = config_tmp.name

        if config_tmp_name is not None:
            model = owl2vec_star.extract_owl2vec_model(ontology_file, config_tmp_name,  uri_doc=None, lit_doc=None, mix_doc=None)
            if output_folder is not None:
                model.save(output_folder)
        return model
