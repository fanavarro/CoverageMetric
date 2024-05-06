import pathlib

import gensim
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment

from BertGeneralModel.BertNounPhraseModel import BertNounPhraseModel
from EmbeddingAlgorithms.Owl2VecStarWrapper import Owl2VecStarWrapper
from results.CoverageMetricResult import Model
from similarity_metrics import SimilarityMetrics

ORDERING_PENALIZATION_FACTOR = 0.25
SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0
bert_noun_phrase_model = BertNounPhraseModel()
def get_pairs_and_score(candidate_solution, levenshtein_similarities):
    pairs = []
    score = 0
    for i in range(0, len(candidate_solution)):
        j = candidate_solution[i]
        token_i = levenshtein_similarities.index[i]
        token_j = levenshtein_similarities.columns[j]
        score = score + levenshtein_similarities.iloc[i, j]
        # score = score + levenshtein_similarities.at[token_i, token_j]
        pairs.append([token_i, token_j])
    return pairs, score



def get_levenshtein_similarity_matrix (a:str, b:str) -> DataFrame:
    tokens_a = a.split(' ')
    tokens_b = b.split(' ')
    # Get matrix with levenshtein distances between the tokens from a and the tokens from b
    levenshtein_similarities = pd.DataFrame(index=tokens_a, columns=tokens_b, dtype='int')
    for token_a in tokens_a:
        for token_b in tokens_b:
            levenshtein_similarities.at[token_a, token_b] = fuzz.ratio(token_a, token_b)

    # We keep at the index the shortest string
    if len(levenshtein_similarities.index) < len(levenshtein_similarities.columns):
        return levenshtein_similarities
    else:
        return levenshtein_similarities.transpose()


def get_ordering_penalization(x):
    if len(x) < 2:
        return 1.0

    unsorted_count = 0
    for i in range(len(x) - 1):
        if x[i] != x[i + 1] - 1:
            # print(f'{x[i]} -> {x[i+1]}')
            unsorted_count = unsorted_count + 1

    return (1 - (unsorted_count * ORDERING_PENALIZATION_FACTOR / len(x)))


def get_lexical_similarity(a: str, b: str) -> float:
    levenshtein_similarity_matrix = get_levenshtein_similarity_matrix(a, b)
    row_ind, col_ind = linear_sum_assignment(levenshtein_similarity_matrix.to_numpy(), maximize=True)
    pairs, score = get_pairs_and_score(col_ind, levenshtein_similarity_matrix)
    # print(f"Best solution: {pairs}; score = {score}; {col_ind}")
    ordering_penalization = get_ordering_penalization(col_ind)
    score_percentage = (float(score) / float(len(levenshtein_similarity_matrix.columns))) * float(ordering_penalization)
    return float(score_percentage/100.0)



def get_semantic_similarity(a: str, b: str, model_a: dict = None, model_b: dict = None) -> float:
    if model_a is None or model_b is None:
        return float(0.0)

    neighbors_a = model_a.get(a)
    neighbors_b = model_b.get(b)
    return bert_noun_phrase_model.get_similarity(neighbors_a, neighbors_b)
    #return SimilarityMetrics.jaccard_similarity(neighbors_a, neighbors_b)



def get_similarity(a: str, b: str, model_a: dict = None, model_b: dict = None,  case_sensitive=True):
    if case_sensitive:
        a = a.lower()
        b = b.lower()
    if model_a is None or model_b is None:
        return get_lexical_similarity(a, b)
    else:
        lexical_similarity = get_lexical_similarity(a, b)
        semantic_similarity = get_semantic_similarity(a, b, model_a, model_b)
        # Weighted arithmetic mean
        similarity = ((lexical_similarity * LEXICAL_WEIGHT) + (semantic_similarity * SEMANTIC_WEIGHT))/(LEXICAL_WEIGHT + SEMANTIC_WEIGHT)
        #print(f"lexical similarity ({a}, {b}) = {lexical_similarity}")
        #print(f"semantic similarity ({a}, {b}) = {semantic_similarity}")
        #print(f"similarity ({a}, {b}) = {similarity}")
        return lexical_similarity, semantic_similarity, similarity


if __name__ == '__main__':
    x = get_lexical_similarity('diabetes mellitus', 'diabetes mellitus')
    x = get_lexical_similarity('diabetes mellitus', 'diabetis melitus')
    x = get_lexical_similarity('diabetes mellitus', 'melitus diabetis')
    x = get_lexical_similarity('diabetes mellitus', 'diabetes type I')
    get_lexical_similarity('a b c', 'a b c')
    get_lexical_similarity('a b c', 'a b d c')
    get_lexical_similarity('a b c', 'a b c d')
    x = get_lexical_similarity('diabetes mellitus', 'diabetis melitus')
    x = get_lexical_similarity('diabetes mellitus', 'mellitus diabetes')
    x = get_lexical_similarity('cabretes mellados', 'diabetes mellitus')
    get_lexical_similarity('a b c', 'd e f')

    ontology_file_path = '/home/fabad/test_embed_comp/go.owl'
    output_embeddings_folder = 'go_embeddings'
    emb_model = Model(pathlib.Path(ontology_file_path).name, Owl2VecStarWrapper().generate_embedding_model(ontology_file_path, output_folder=output_embeddings_folder))

    #emb_model = gensim.models.Word2Vec.load(output_embeddings_folder)
    get_similarity('hexose metabolic process','glucose metabolic process', emb_model)
    get_similarity('hexose catabolic process', 'glucose metabolic process', emb_model)



