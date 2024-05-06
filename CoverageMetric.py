import argparse
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import torch
from gensim.models import KeyedVectors

from EmbeddingAlgorithms.Doc2VecWrapper import Doc2VecWrapper
from EmbeddingAlgorithms.FastTextWrapper import FastTextWrapper
from EmbeddingAlgorithms.Owl2VecStarWrapper import Owl2VecStarWrapper
from EmbeddingAlgorithms.Word2VecWarapper import Word2VecWrapper
from nlp_utils import NounPhraseExtractor
from nlp_utils.OntologyPreprocessing import get_noun_phrases_and_normalize_annotations
from results.CoverageMetricResult import Model
from results.NormalizedNounPhrases import NormalizedNounPhrases
from owlready2 import *
from datetime import datetime
import ctypes

from similarity_metrics import SimilarityMetrics, CustomStringSimilarity

NORMALIZED_TEXT_FOLDER = 'normalized_text'

TOP_NEIGHBORS = 10

annotation_properties_to_consider = ['http://www.w3.org/2000/01/rdf-schema#label',
                                     'http://www.w3.org/2004/02/skos/core#prefLabel',
                                     'http://www.w3.org/2004/02/skos/core#altLabel',
                                     'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
                                     'http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
                                     'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym',
                                     'http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym']


def get_class_annotations(owl_class):
    annotations = {}
    for prop in owl_class.get_properties(owl_class):
        if prop.iri in annotation_properties_to_consider:
            # annotations[prop] = prop[owl_class]
            annotations[prop] = list(set(prop[owl_class]))
    return annotations

def get_best_owl_class_property_for_noun_phrase(owl_class, noun_phrase: str, score_func, model_ontology = None, model_text = None, case_sensitive = True):
    best_class_annotation = []
    best_score = float('-inf')
    best_lexical_similarity = []
    best_semantic_similarity = []
    best_ontology_neighbors = []
    best_text_neighbors = []

    class_annotations = get_class_annotations(owl_class)
    for property, annotations in class_annotations.items():
        for annotation_value in annotations:
            if not isinstance(annotation_value, str):
                continue

            if annotation_value not in model_ontology.keys():
                continue

            lexical_similarity, semantic_similarity, score = score_func(annotation_value, noun_phrase, model_ontology, model_text, case_sensitive)
            if score > best_score:
                best_class_annotation = []
                best_class_annotation.append({'property': property.iri, 'value': annotation_value})

                best_score = score
                best_lexical_similarity = [lexical_similarity]
                best_semantic_similarity = [semantic_similarity]
                best_ontology_neighbors = [model_ontology.get(annotation_value)]
                best_text_neighbors = [model_text.get(noun_phrase)]

            elif score == best_score and len(best_class_annotation) != 0:
                if annotation_properties_to_consider.index(best_class_annotation[0].get('property')) > annotation_properties_to_consider.index(property.iri):
                    best_class_annotation = [{'property': property.iri, 'value': annotation_value}]
                    best_score = score
                    best_lexical_similarity = [lexical_similarity]
                    best_semantic_similarity = [semantic_similarity]
                    best_ontology_neighbors = [model_ontology.get(annotation_value)]
                    best_text_neighbors = [model_text.get(noun_phrase)]

                elif annotation_properties_to_consider.index(best_class_annotation[0].get('property')) == annotation_properties_to_consider.index(property.iri):
                    best_class_annotation.append({'property': property.iri, 'value': annotation_value})
                    best_lexical_similarity.append(lexical_similarity)
                    best_semantic_similarity.append(semantic_similarity)
                    best_ontology_neighbors.append(model_ontology.get(annotation_value))
                    best_text_neighbors.append(model_text.get(noun_phrase))
    return {'noun_phrase': noun_phrase,
            'OWLClass': owl_class.iri,
            'best_class_annotation': best_class_annotation,
            'best_lexical_similarity': best_lexical_similarity,
            'best_semantic_similarity': best_semantic_similarity,
            'best_score': best_score,
            'best_ontology_neighbors': best_ontology_neighbors,
            'best_text_neighbors': best_text_neighbors}

def get_best_noun_phrase_match_for_class(ontology_address, owl_class_iri, normalized_noun_phrases_address, score_func, i, n, model_ontology_address = None, model_text_address = None, case_sensitive = True):
    print(f"Processing class {i}/{n} ({owl_class_iri})")
    normalized_noun_phrases = ctypes.cast(normalized_noun_phrases_address, ctypes.py_object).value
    ontology = ctypes.cast(ontology_address, ctypes.py_object).value
    owl_class = ontology.search_one(iri=owl_class_iri)
    model_ontology = ctypes.cast(model_ontology_address, ctypes.py_object).value if model_ontology_address is not None else None
    model_text = ctypes.cast(model_text_address, ctypes.py_object).value if model_text_address is not None else None
    best_scores = [{'best_score': float('-inf')}]
    for noun_phrase in normalized_noun_phrases.normalizedNounPhrases.values():
        class_score = get_best_owl_class_property_for_noun_phrase(owl_class, noun_phrase, score_func, model_ontology, model_text, case_sensitive)
        if class_score.get('best_score') > best_scores[0].get('best_score'):
            best_scores = [class_score]
        elif class_score.get('best_score') == best_scores[0].get('best_score'):
            if class_score.get('best_score') == float('-inf'): # The class does not have any annotation...
                best_scores = [class_score]
            elif annotation_properties_to_consider.index(best_scores[0].get('best_class_annotation')[0].get('property')) > annotation_properties_to_consider.index(class_score.get('best_class_annotation')[0].get('property')):
                best_scores = [class_score]
            elif annotation_properties_to_consider.index(best_scores[0].get('best_class_annotation')[0].get('property')) == annotation_properties_to_consider.index(class_score.get('best_class_annotation')[0].get('property')):
                best_scores.append(class_score)

    return best_scores



def match_ontology_with_noun_phrases(ontology, normalized_noun_phrases, score_function, output_file, threads:int, model_ontology:KeyedVectors = None, model_text:KeyedVectors = None, case_sensitive = True):
    best_matches_for_classes_futures = {}
    ontology_classes = list(ontology.classes())
    i = 0
    n = len(ontology_classes)
    ontology_address = id(ontology)
    model_ontology_address = id(model_ontology)
    model_text_address = id(model_text)
    normalized_noun_phrases_address = id(normalized_noun_phrases)
    executor = ProcessPoolExecutor(max_workers=threads)

    for owl_class in ontology_classes:
        i = i + 1
        best_matches_for_classes_futures[owl_class] = executor.submit(get_best_noun_phrase_match_for_class, ontology_address, owl_class.iri, normalized_noun_phrases_address, score_function, i, n, model_ontology_address, model_text_address, case_sensitive)

    executor.shutdown(wait=True)

    with open(output_file, "w+") as file:
        # for noun_phrase, normalized_noun_phrase in normalized_noun_phrases.normalizedNounPhrases.items():
        #    file.write(f"{noun_phrase.text}  ->  {normalized_noun_phrase}  ->  {normalized_noun_phrases.get_frequency(normalized_noun_phrase)}\n")
        file.write("Class_IRI\tAnnotation_property\tAnnotation_value\tNoun_phrase\tLexical_similarity\tSemantic_similarity\tScore\tOntology_neighbors\tText_neighbors\n")
        for owl_class, best_matches_future in best_matches_for_classes_futures.items():
            best_matches = best_matches_future.result()
            for best_match in best_matches:
                owl_class_iri = best_match.get('OWLClass')
                noun_phrase = best_match.get('noun_phrase')
                score = best_match.get('best_score')
                if len(best_match.get('best_class_annotation')) > 0: # Maybe a class does not have any annotation, thus scores cannot be computed
                    for i in range(len(best_match.get('best_class_annotation'))):
                        property_iri = best_match.get('best_class_annotation')[i].get('property')
                        property_value = best_match.get('best_class_annotation')[i].get('value')
                        semantic_score = best_match.get('best_semantic_similarity')[i]
                        lexical_score = best_match.get('best_lexical_similarity')[i]
                        ontology_neighbors = ' | '.join(best_match.get('best_ontology_neighbors')[i])
                        text_neighbors = ' | '.join(best_match.get('best_text_neighbors')[i])
                        file.write(f"{owl_class_iri}\t{property_iri}\t{property_value}\t{noun_phrase}\t{lexical_score}\t{semantic_score}\t{score}\t{ontology_neighbors}\t{text_neighbors}\n")
                else:
                    property_iri = ''
                    property_value = ''
                    semantic_score = float('-inf')
                    lexical_score = float('-inf')
                    ontology_neighbors = ''
                    text_neighbors = ''
                    file.write(f"{owl_class_iri}\t{property_iri}\t{property_value}\t{noun_phrase}\t{lexical_score}\t{semantic_score}\t{score}\t{ontology_neighbors}\t{text_neighbors}\n")


def get_best_class_match_for_noun_phrase(ontology, noun_phrase: str, score_func, model_ontology:dict = None, model_text:dict = None, case_sensitive = True):
    best_class = []
    best_class_annotation = []
    best_score = float('-inf')
    best_lexical_similarity = []
    best_semantic_similarity = []
    best_ontology_neighbors = []
    best_text_neighbors = []

    for owl_class in ontology.classes():
        class_annotations = get_class_annotations(owl_class)
        for property, annotations in class_annotations.items():
            for owl_class_annotation in annotations:
                if not isinstance(owl_class_annotation, str):
                    continue

                if owl_class_annotation not in model_ontology.keys():
                    continue

                lexical_similarity, semantic_similarity, score = score_func(owl_class_annotation, noun_phrase, model_ontology, model_text, case_sensitive)
                if score > best_score:
                    best_score = score
                    best_class = []
                    best_class_annotation = []
                    best_lexical_similarity = []
                    best_semantic_similarity = []
                    best_ontology_neighbors = []
                    best_text_neighbors = []

                    best_class.append(owl_class)
                    best_class_annotation.append({'property': property,
                                             'value': owl_class_annotation})
                    best_lexical_similarity.append(lexical_similarity)
                    best_semantic_similarity.append(semantic_similarity)
                    best_ontology_neighbors.append(model_ontology.get(owl_class_annotation))
                    best_text_neighbors.append(model_text.get(noun_phrase))

                elif score == best_score and len(best_class_annotation) != 0:
                    if annotation_properties_to_consider.index(best_class_annotation[0].get('property').iri) > annotation_properties_to_consider.index(property.iri):
                        best_class = []
                        best_class_annotation = []
                        best_lexical_similarity = []
                        best_semantic_similarity = []
                        best_ontology_neighbors = []
                        best_text_neighbors = []

                        best_class.append(owl_class)
                        best_class_annotation.append({'property': property, 'value': owl_class_annotation})
                        best_score = score
                        best_lexical_similarity.append(lexical_similarity)
                        best_semantic_similarity.append(semantic_similarity)
                        best_ontology_neighbors.append(model_ontology.get(owl_class_annotation))
                        best_text_neighbors.append(model_text.get(noun_phrase))
                    elif annotation_properties_to_consider.index(best_class_annotation[0].get('property').iri) == annotation_properties_to_consider.index(property.iri):
                        best_class.append(owl_class)
                        best_class_annotation.append({'property': property, 'value': owl_class_annotation})
                        best_lexical_similarity.append(lexical_similarity)
                        best_semantic_similarity.append(semantic_similarity)
                        best_ontology_neighbors.append(model_ontology.get(owl_class_annotation))
                        best_text_neighbors.append(model_text.get(noun_phrase))

    # Convert to plain objects
    best_class_iris = [owl_class.iri for owl_class in best_class]
    best_class_annotation_plain = []
    for x in best_class_annotation:
        best_class_annotation_plain.append({'property': x.get('property').iri, 'value': x.get('value')})
    return {'best_class': best_class_iris,
            'best_class_annotation': best_class_annotation_plain,
            'noun_phrase': noun_phrase,
            'best_lexical_similarity': best_lexical_similarity,
            'best_semantic_similarity': best_semantic_similarity,
            'best_score': best_score,
            'best_ontology_neighbors': best_ontology_neighbors,
            'best_text_neighbors': best_text_neighbors}

# Test for parallelization
def get_best_class_match_for_noun_phrase_verbose(ontology_address, model_ontology_address, model_text_address, normalized_noun_phrase, score_function, i, n, case_sensitive = True):
    ontology = ctypes.cast(ontology_address, ctypes.py_object).value
    model_ontology = ctypes.cast(model_ontology_address, ctypes.py_object).value
    model_text = ctypes.cast(model_text_address, ctypes.py_object).value

    print(f"Processing noun phrase '{normalized_noun_phrase}' ({i}/{n}) {datetime.fromtimestamp(time.time())}")
    return get_best_class_match_for_noun_phrase(ontology, normalized_noun_phrase, score_function, model_ontology, model_text, case_sensitive)



def match_noun_phrases_with_ontology(ontology, normalized_noun_phrases, score_function, output_file, threads:int, model_ontology:dict = None, model_text:dict = None, case_sensitive = True):
    best_matches_for_noun_phrases_futures = {}
    i = 0
    n = len(normalized_noun_phrases.normalizedNounPhrases.items())
    ontology_address = id(ontology)
    model_ontology_address = id(model_ontology)
    model_text_address = id(model_text)
    executor = ProcessPoolExecutor(max_workers=threads)
    for noun_phrase, normalized_noun_phrase in list(normalized_noun_phrases.normalizedNounPhrases.items()):
        i = i + 1
        best_matches_for_noun_phrases_futures[normalized_noun_phrase] = executor.submit(get_best_class_match_for_noun_phrase_verbose, ontology_address, model_ontology_address, model_text_address, normalized_noun_phrase, score_function, i, n, case_sensitive)
    executor.shutdown(wait=True)

    with open(output_file, "w+") as file:
        file.write("Noun_phrase\tClass_IRI\tAnnotation_property\tAnnotation_value\tLexical_similarity\tSemantic_similarity\tScore\tText_neighbors\tOntology_neighbors\n")
        for noun_phrase, best_match_future in best_matches_for_noun_phrases_futures.items():
            best_match = best_match_future.result()
            for i in range(0, len(best_match['best_class'])):
                best_class = best_match['best_class'][i]
                best_class_annotation = best_match['best_class_annotation'][i]
                property = best_class_annotation ['property']
                annotation = best_class_annotation['value']
                lexical_similarity = best_match['best_lexical_similarity'][i]
                semantic_similarity = best_match['best_semantic_similarity'][i]
                score = best_match['best_score']
                text_neighbors = " | ".join(best_match['best_text_neighbors'][i])
                ontology_neighbors = " | ".join(best_match['best_ontology_neighbors'][i])
                file.write(
                    f"{best_match['noun_phrase']}\t{best_class}\t{property}\t{annotation}\t{lexical_similarity}\t{semantic_similarity}\t{score}\t{text_neighbors}\t{ontology_neighbors}\n")

def get_normalized_noun_phrases_and_normalize_text(text_folder_path, to_lower_case=False, only_keywords=False) -> NormalizedNounPhrases:
    normalized_text_folder_path = pathlib.Path(text_folder_path, NORMALIZED_TEXT_FOLDER)
    normalized_noun_phrases: NormalizedNounPhrases = NormalizedNounPhrases()
    for file_path in text_folder_path.iterdir():
        if not file_path.is_file():
            continue
        with open(file_path, "r") as file:
            text = file.read()
            normalized_noun_phrases.merge(NounPhraseExtractor.get_normalized_noun_phrases(text, only_keywords=only_keywords, to_lower_case=to_lower_case))
            NounPhraseExtractor.normalize_text_to_file(text, normalized_noun_phrases, pathlib.Path(normalized_text_folder_path, file_path.name))
    return normalized_noun_phrases


def get_normalized_noun_phrases(text_folder_path, to_lower_case=False, only_keywords=False) -> NormalizedNounPhrases:
    normalized_noun_phrases: NormalizedNounPhrases = NormalizedNounPhrases()
    for file_path in text_folder_path.iterdir():
        if not file_path.is_file():
            continue
        with open(file_path, "r") as file:
            text = file.read()
            normalized_noun_phrases.merge(NounPhraseExtractor.get_normalized_noun_phrases(text, only_keywords=only_keywords, to_lower_case=to_lower_case))
    return normalized_noun_phrases


def keep_only(model: Model, new_keys: list) -> KeyedVectors:
    new_keyed_vectors = KeyedVectors(model.embeddings_model.wv.vectors.shape[1])
    new_vectors = []
    for key in new_keys:
        new_vectors.append(model.embeddings_model.wv[key])
        #for token in key.split(' '):
        #    new_vectors.append(model.embeddings_model.wv[token])

    if len(new_keys) != 0:
        new_keyed_vectors.add_vectors(new_keys, new_vectors)
        return new_keyed_vectors
    return None



def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_folder', action='store', required=True, help='Folder containing natural language text'
                                                                              ' files.')
    parser.add_argument('--ontology', action='store', required=True, help='Ontology file.')
    parser.add_argument('--term_freq_threshold', action='store', type=float, required=False, default=0, help='Threshold to filter the detected noun phrases in the free text based on the noun phrase frequency in the corpus. This threshold is based on the normalized frequency of each term (term frequency / max frequency found), that is from 0 to 1.')
    parser.add_argument('--output_prefix', action='store', required=True, help='Output prefix to store the results')
    parser.add_argument('--threads', action='store', type=int, required=False, default=1, help='Threads to use')
    return parser



def get_ontology_neighbors_cache(ontology, model: KeyedVectors) -> dict:
    ontology_classes = list(ontology.classes())
    count = 0
    n = len(ontology_classes)
    neighbors_cache = {}
    for owl_class in ontology_classes:
        count = count + 1
        print(f'Getting neighbors from class {owl_class} ({count}/{n})')
        owl_class_annotations = get_class_annotations(owl_class) # map annotation_iri -> list of annotation_values
        for property_iri, annotation_values in owl_class_annotations.items():
            for annotation_value in annotation_values:
                if not model.has_index_for(annotation_value):
                    print(f"Key '{annotation_value}' not present in vocabulary. Setting empty set as neighbors and skipping...")
                    neighbors = tuple()
                else:
                    neighbors = [x[0] for x in model.similar_by_word(annotation_value, topn=TOP_NEIGHBORS, restrict_vocab=None)]
                neighbors_cache[annotation_value] = tuple(neighbors)
    return neighbors_cache



def get_noun_phrase_neighbors_cache(noun_phrases, model: KeyedVectors) -> dict:
    neighbors_cache = {}
    for noun_phrase, normalized_noun_phrase in list(noun_phrases.normalizedNounPhrases.items()):
        neighbors = [x[0] for x in model.similar_by_word(normalized_noun_phrase, topn=TOP_NEIGHBORS, restrict_vocab=None)]
        neighbors_cache[normalized_noun_phrase] = tuple(neighbors)

    return neighbors_cache


def get_normalized_ontology_path(ontology_path: pathlib.Path):
    return pathlib.Path(ontology_path.parent, ontology_path.stem + "_processed" + ontology_path.suffix)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    threads = args.threads
    case_sensitive = False
    noun_phrases_to_lower_case = True
    score_function = SimilarityMetrics.custom_string_similarity
    torch.set_num_threads(1) # to prevent hangs when each thread is using bert model

    # Ontology
    ontology_file_path = pathlib.Path(args.ontology)
    normalized_ontology_file_path = get_normalized_ontology_path(ontology_file_path)
    normalized_noun_phrases_ontology: NormalizedNounPhrases = get_noun_phrases_and_normalize_annotations(ontology_file_path, normalized_ontology_file_path, noun_phrases_to_lower_case)
    normalized_noun_phrases_ontology_filtered = normalized_noun_phrases_ontology.filter_by_normalized_frequency(args.term_freq_threshold)
    ontology_emb_model = Model(pathlib.Path(normalized_ontology_file_path).name,
                               Owl2VecStarWrapper().generate_embedding_model(normalized_ontology_file_path))
    ontology_emb_wv = keep_only(ontology_emb_model, list(normalized_noun_phrases_ontology_filtered.normalizedNounPhrases.values()))
    ontology = get_ontology(f"file://{str(normalized_ontology_file_path)}").load()
    #ontology_neighbors_cache = get_ontology_neighbors_cache(ontology, ontology_emb_wv)
    ontology_neighbors_cache = get_noun_phrase_neighbors_cache(normalized_noun_phrases_ontology_filtered, ontology_emb_wv)

    # Free text
    text_folder_path = pathlib.Path(args.text_folder)
    normalized_text_folder_path = pathlib.Path(text_folder_path, NORMALIZED_TEXT_FOLDER)
    normalized_noun_phrases: NormalizedNounPhrases = get_normalized_noun_phrases_and_normalize_text(text_folder_path, noun_phrases_to_lower_case, only_keywords=False)
    normalized_noun_phrases_filtered = normalized_noun_phrases.filter_by_normalized_frequency(args.term_freq_threshold)
    text_emb_model = Model(pathlib.Path(text_folder_path).name, FastTextWrapper().generate_embedding_model(normalized_text_folder_path))
    text_emb_wv = keep_only(text_emb_model, list(normalized_noun_phrases_filtered.normalizedNounPhrases.values()))
    text_neighbors_cache = get_noun_phrase_neighbors_cache(normalized_noun_phrases_filtered, text_emb_wv)

    output_file = args.output_prefix + 'text2class.tsv'
    os.makedirs(pathlib.Path(output_file).parent, exist_ok=True)
    match_noun_phrases_with_ontology(ontology, normalized_noun_phrases_filtered, score_function, output_file, threads, ontology_neighbors_cache, text_neighbors_cache, case_sensitive)


    output_file = args.output_prefix + 'class2text.tsv'
    os.makedirs(pathlib.Path(output_file).parent, exist_ok=True)
    match_ontology_with_noun_phrases(ontology, normalized_noun_phrases_filtered, score_function, output_file, threads, ontology_neighbors_cache, text_neighbors_cache, case_sensitive)
