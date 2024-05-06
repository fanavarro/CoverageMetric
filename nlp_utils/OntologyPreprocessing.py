from typing import Literal, get_origin

from owlready2 import *
from nlp_utils.NounPhraseExtractor import get_normalized_noun_phrases, normalize_text, NormalizedNounPhrases

DESCRIPTION_PROPERTIES = ['http://purl.obolibrary.org/obo/IAO_0000115',
                          'http://www.w3.org/2004/02/skos/core#definition',
                          'http://www.w3.org/2000/01/rdf-schema#comment',
                          'http://purl.org/dc/elements/1.1/description',
                          'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97']


def remove_annotations(subject: str, predicate: str):
    delete_query = f'''
           DELETE {{ ?s ?p ?o }} 
           WHERE
           {{
                VALUES ?s {{ <{subject}> }} .
                VALUES ?p {{ <{predicate}> }} .
                ?s ?p ?o .
           }}
    '''
    default_world.sparql(delete_query)


def add_annotations(ontology: Ontology, subject: str, predicate: str, annotation_values: list):
    for annotation_value in annotation_values:
        add_query = f'''
               INSERT {{ ?s ?p ?o }} 
               WHERE
               {{
                    BIND (<{subject}> AS ?s) .
                    BIND (<{predicate}> AS ?p) .
                    BIND ("{str(annotation_value)}" AS ?o) .
               }}
        '''
        with ontology:
            default_world.sparql(add_query)


def get_noun_phrases_and_normalize_annotations(input_ontology_path, output_ontology_path, to_lower_case = False):
    ontology = get_ontology(f"file://{str(input_ontology_path)}").load()
    normalized_noun_phrases: NormalizedNounPhrases = NormalizedNounPhrases()
    for owl_class in ontology.classes():
        new_annotations = dict()
        for prop in owl_class.get_properties(owl_class):
            annotation_values = [str(annotation_value) for annotation_value in list(set(prop[owl_class])) if isinstance(annotation_value, str)]
            if annotation_values:
                normalized_noun_phrases_annotation, normalized_texts = get_noun_phrases_and_normalize_text(annotation_values, to_lower_case)
                normalized_noun_phrases.merge(normalized_noun_phrases_annotation)
                new_annotations[prop] = normalized_texts

        for prop, annotation_values in new_annotations.items():
            remove_annotations(owl_class.iri, prop.iri)
            add_annotations(ontology, owl_class.iri, prop.iri, annotation_values)

    ontology.save(file=str(output_ontology_path), format="rdfxml")
    return normalized_noun_phrases


def get_noun_phrases_and_normalize_text(texts: list, to_lower_case=False):
    normalized_texts = []
    for text in texts:
        normalized_noun_phrases = get_normalized_noun_phrases(text, to_lower_case=to_lower_case, only_keywords=False)
        normalized_text = normalize_text(text, normalized_noun_phrases)
        normalized_texts.append(normalized_text)
    return normalized_noun_phrases, normalized_texts
