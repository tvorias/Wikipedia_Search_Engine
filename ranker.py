"""
This is the template for implementing the rankers for your search engine.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import math


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # This is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        if isinstance(scorer, type):
            scorer = scorer(index)
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        # TODO: Tokenize the query and remove stopwords, if needed

        # TODO If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.

        # TODO Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers) which is ok and totally expected.

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #  a document ID to a dictionary of the counts of the query terms in that document.
        #  You will pass the dictionary to the RelevanceScorer as input.

        # TODO: Rank the documents using a RelevanceScorer

        # TODO: Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        
        # Tokenize the query and remove stopwords
        query_tokens = self.tokenize(query)
        query_tokens = [token if token.lower() not in self.stopwords else None for token in query_tokens]
        query_word_counts = Counter(query_tokens)
        
        doc_term_counts = defaultdict(Counter)

        for token in set(query_tokens):
            if token:
                #print(self.index.get_postings(token))
                postings = self.index.get_postings(token)
                for docid, frequency in postings:
                    doc_term_counts[docid][token] = frequency

        results = []
        for doc_id, term_counts in doc_term_counts.items():
            score = self.scorer.score(doc_id, doc_word_counts=term_counts, query_word_counts=query_word_counts)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)

        if pseudofeedback_num_docs > 0:
            results_top = results[:pseudofeedback_num_docs]

            pseudo_word_counts = Counter()
            for docid, score in results_top:
                text = self.raw_text_dict[docid]
                tokens = self.tokenize(text)
                tokens = [token for token in tokens if token not in self.stopwords]
                pseudo_word_counts.update(tokens)
            
            all_terms = set()
            for term, frequency in query_word_counts.items():
                all_terms.add(term)
            for term, frequency in pseudo_word_counts.items():
                all_terms.add(term)

            result_query_counts = {}
            for term in all_terms:
                result_query_counts[term] = (query_word_counts[term] * pseudofeedback_alpha) + (pseudofeedback_beta/pseudofeedback_num_docs) * pseudo_word_counts[term]

            updated_query = set()
            for term in result_query_counts.keys():
                updated_query.add(term)

            doc_term_counts = defaultdict(Counter)

            for token in updated_query:
                if token:
                    postings = self.index.get_postings(token)
                    for docid, frequency in postings:
                        doc_term_counts[docid][token] = frequency

            results_pseudo = []
            for doc_id, term_counts in doc_term_counts.items():
                score = self.scorer.score(doc_id, doc_word_counts=term_counts, query_word_counts=result_query_counts)
                results_pseudo.append((doc_id, score))

            results_pseudo.sort(key=lambda x: x[1], reverse=True)

            return results_pseudo
        else:
            return results

class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """

    def __init__(self, index: InvertedIndex, parameters) -> None:
        self.index = index
        self.parameters = parameters
        self.statistics = index.get_statistics()
        self.mean_document_length = self.statistics['mean_document_length']
        self.total_documents = self.statistics['number_of_documents']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError


# Starter code for implementing unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        return score 


# Starter code for implementing DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        return score

    
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        # self.b = parameters['b']
        # self.k1 = parameters['k1']
        # self.k3 = parameters['k3']
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        score = 0
        b = self.parameters['b']
        k1 = self.parameters['k1']
        k3 = self.parameters['k3']


        for term, frequency in query_word_counts.items():
            if term == None or term not in doc_word_counts:
                continue

            indoc_freq = doc_word_counts[term]

            if indoc_freq == 0:
                continue

            postings = self.index.get_postings(term)
            if postings is None:
                continue

            docs_with_term = len(postings)
            variant_IDF = math.log((self.index.get_statistics()['number_of_documents'] - docs_with_term + 0.5)/(docs_with_term + 0.5))
            document_length = self.index.get_doc_metadata(docid)['length']
            variant_TF = ((k1 + 1) * indoc_freq)/(k1 * (1 - b + b * (document_length/self.index.get_statistics()['mean_document_length'])) + indoc_freq)
            normalized_QTF = ((k3 + 1) * frequency)/(k3 + frequency)
            query_score = variant_IDF * variant_TF * normalized_QTF

            score += query_score
        return score


class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25
        
        score = 0
        b = self.parameters['b']
        k1 = self.parameters['k1']
        k3 = self.parameters['k3']

        # get the number of seed documents
        
        big_R = self.relevant_doc_index.get_statistics()['number_of_documents']  # number of relevant documents 

        for query, frequency in query_word_counts.items():
            
            r_i = len(self.relevant_doc_index.get_postings(query))

            if query == None or query not in doc_word_counts:
                continue

            indoc_freq = doc_word_counts[query]

            if indoc_freq == 0:
                continue

            postings = self.index.get_postings(query)
            if postings is None:
                continue

            docs_with_term = len(postings)  # number of documents containing the term
            
            variant_IDF = math.log(((r_i  + 0.5) * (self.index.get_statistics()['number_of_documents'] - docs_with_term - big_R + r_i  + 0.5)) / ((docs_with_term - r_i  + 0.5) * (big_R - r_i  + 0.5)))
            document_length = self.index.get_doc_metadata(docid)['length']
            variant_TF = ((k1 + 1) * indoc_freq)/(k1 * ((1 - b) + b * (document_length/self.index.get_statistics()['mean_document_length'])) + indoc_freq)
            normalized_QTF = ((k3 + 1) * frequency)/(k3 + frequency)
            
            query_score = variant_IDF * variant_TF * normalized_QTF

            score += query_score
        return score


class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        score = 0
        
        for query, qtf in query_word_counts.items():
            if query == None:
                continue

            postings = self.index.get_postings(query)
            if postings == None:
                continue

            indoc_freq = doc_word_counts.get(query, 0)
            if indoc_freq == 0:
                continue

            idf = math.log((self.statistics['number_of_documents'] + 1) / len(postings))
            document_length = self.index.get_doc_metadata(docid)['length']

            tf = (1 + math.log(1 + math.log(indoc_freq))) / (1 - self.b + (self.b * (document_length/self.mean_document_length)))

            score += (qtf * tf * idf)

        return score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0

        for query in query_word_counts.keys():
            if query == None or query not in doc_word_counts:
                continue

            indoc_freq = doc_word_counts.get(query, 0)
            if indoc_freq == 0:
                continue

            postings = self.index.get_postings(query)

            if postings == None:
                continue

            docs_with_term = len(postings)
            idf = 1 + (math.log(self.statistics['number_of_documents']/docs_with_term))
            tf = math.log(indoc_freq + 1)

            score += (tf * idf)
        return score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        # TODO: Save any new arguments that are needed as fields of this class

        self.raw_text_dict = raw_text_dict
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, max_length=500)


    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle edge cases
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)

        if docid not in self.raw_text_dict or not query:
            return 0.0
        document_text = self.raw_text_dict[docid]

        score = self.cross_encoder.predict([(query, document_text)])

        return score[0]


# TODO: Implement your own ranker with proper heuristics
#  You can use the Ranker class as a template. My ranker creates a score based on the number of query terms in the document.
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        super().__init__(index, parameters)
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]):
        
        score = 0
        for query, qtf in query_word_counts.items():
            indoc_freq = doc_word_counts.get(query, 0)
            if indoc_freq > 0:
                score += 1
        if score == 0:
            return score
        else:
            score = score/len(query_word_counts.keys())

        return score


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        # Print randomly ranked results
        return 10

