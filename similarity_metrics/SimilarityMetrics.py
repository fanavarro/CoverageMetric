import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors

from results.CoverageMetricResult import Model
from similarity_metrics import CustomStringSimilarity


def get_average_euclidean_distance_between_keys(keyed_vectors_a, keyed_vectors_b):
    """
    Compute the average euclidean distance between two sets of keyed vectors. Both keyed_vectors_a and keyed_vectors_b
    must have the same keys.
    :param keyed_vectors_a:
    :param keyed_vectors_b:
    :return:
    """
    total_distance = 0.0
    for key in keyed_vectors_a.key_to_index.keys():
        total_distance = total_distance + np.linalg.norm(keyed_vectors_a[key] - keyed_vectors_b[key])
    return total_distance/len(keyed_vectors_a.key_to_index.keys())


def cosine_similarity(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    #dist = 1. - similiarity
    return similiarity


def get_cosine_similarity(keyed_vectors_a, keyed_vectors_b):
    """
    Compute the cosine similarity between each common key pair in keyed_vectors_a and keyed_vectors_b
    :param keyed_vectors_a:
    :param keyed_vectors_b:
    :return: A list of dictionaries with key and cos_sim fields, sorted by cos_sim
    """
    similarities = []
    for key in keyed_vectors_a.key_to_index.keys():
        similarities.append({'key': key, 'cos_sim': cosine_similarity(keyed_vectors_a[key], keyed_vectors_b[key])})
    similarities.sort(key=lambda x: x['cos_sim'], reverse=True)
    return similarities


def get_average_cosine_similarity_between_keys(keyed_vectors_a, keyed_vectors_b):
    """
    Get the average cosine similarity between each vector pair from keyed_vectors_a and keyed_vectors_b
    :param keyed_vectors_a:
    :param keyed_vectors_b:
    :return:
    """
    similarities = get_cosine_similarity(keyed_vectors_a, keyed_vectors_b)
    cosine_values = [similarity.get('cos_sim') for similarity in similarities]
    average_cosine_value = sum(cosine_values)/len(cosine_values)
    return average_cosine_value


def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)
    return len(set_a.intersection(set_b))/len(set_a.union(set_b))

def jaccard_similarity_with_repeats(a_list, b_list):
    common_elements = 0
    for a_element in a_list:
        if a_element in b_list:
            common_elements = common_elements + 1
            b_list.remove(a_element)
    total_elements = len(a_list) + len(b_list)
    return float(common_elements)/float(total_elements)


def jaccard_dissimilarity(a, b):
    return 1 - jaccard_similarity(a, b)


def jaccard_similarity_string(a:str, b:str):
    return jaccard_similarity(a.split(' '), b.split(' '))

def jaccard_dissimilarity_string(a:str, b:str):
    return 1 - jaccard_similarity_string(a, b)

def jaccard_similarity_string_with_repeats(a:str, b:str):
    a_list = a.split(' ')
    b_list = b.split(' ')
    return jaccard_similarity_with_repeats(a_list, b_list)


def get_pairs_of_tokens(tokens_a, tokens_b):
    pairs = [[]]
    if len(tokens_a) == 1:
        token_a = tokens_a[0]
        for token_b in tokens_b:
            pairs.append([token_a, token_b])
    elif len(tokens_b) == 1:
        token_b = tokens_b[0]
        for token_a in tokens_a:
            pairs.append([token_a, token_b])
    else:
        for token_a in tokens_a:
            for token_b in tokens_b:
                pairs.append([token_a, token_b])
                get_pairs_of_tokens(list(filter(lambda x: x != token_a, tokens_a)),
                                    list(filter(lambda x: x != token_b, tokens_b)))
    return pairs


def custom_string_similarity(a: str, b: str, model_a: dict = None, model_b: dict = None, case_sensitive=True):
    return CustomStringSimilarity.get_similarity(a, b, model_a, model_b, case_sensitive)





