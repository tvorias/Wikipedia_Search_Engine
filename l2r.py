from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer, YourRanker
import math
from importlib import reload
from vector_ranker import VectorRanker


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """

        self.ranker = ranker
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.feature_extractor = feature_extractor
        self.YourRanker=YourRanker
        
        self.model = LambdaMART()
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        
        if not query_to_document_relevance_scores:
            return [], [], []
        
        X = []
        y = []
        qgroups = []

        # TODO: For each query and the documents that have been rated for relevance to that query,
        #       process these query-document pairs into features

            # TODO: Accumulate the token counts for each document's title and content here

            # TODO: For each of the documents, generate its features, then append
            #       the features and relevance score to the lists to be returned

            # Make sure to keep track of how many scores we have for this query

        for query, doc_rel_scores in tqdm(query_to_document_relevance_scores.items()):
            query_parts = self.document_preprocessor.tokenize(query)
            doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts=query_parts)
            title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts=query_parts)
            for docid, relevance_score in doc_rel_scores:
              
                doc_features = self.feature_extractor.generate_features(docid=docid, doc_word_counts=doc_word_counts[docid], 
                                                                        title_word_counts=title_word_counts[docid], query_parts=query_parts, query=query)
                X.append(doc_features)
                y.append(relevance_score)
        
            qgroups.append(len(doc_rel_scores))
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        #       create a dictionary that keeps track of their counts for the query word
        doc_term_counts = defaultdict(lambda: defaultdict(int))

        for query_token in set(query_parts):
            if query_token == None:
                continue
            else:
                postings = index.get_postings(query_token)
                if postings:
                    for doc_id, term_frequency in postings:
                        doc_term_counts[doc_id][query_token] += int(term_frequency)

        return doc_term_counts

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        
        # TODO: Prepare the training data by featurizing the query-doc pairs and
        #       getting the necessary datastructures
        
        # TODO: Train the model

        training_data = pd.read_csv(training_data_filename)
        query_to_document_relevance_scores = {}

        for index, row in training_data.iterrows():
            query = row['query']
            docid = int(row['docid'])
            rel = int(row['rel'])

            if query in query_to_document_relevance_scores:
                query_to_document_relevance_scores[query].append((docid, rel))
            else:
                query_to_document_relevance_scores[query] = [(docid, rel)]

        X, y, qgroups = self.prepare_training_data(query_to_document_relevance_scores)

        self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # TODO: Return a prediction made using the LambdaMART model
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        predictions = self.model.predict(X)
        return predictions

    # TODO Implement MMR diversification for a given list of documents and their cosine similarity scores
    @staticmethod
    def maximize_mmr(thresholded_search_results: list[tuple[int, float]], similarity_matrix: np.ndarray,
                     list_docs: list[int], mmr_lambda: int) -> list[tuple[int, float]]:
        """
        Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm
        on the list.
        It should return a list of the same length with the same overall documents but with different document ranks.
        
        Args:
            thresholded_search_results: The thresholded search results
            similarity_matrix: Precomputed similarity scores for all the thresholded search results
            list_docs: The list of documents following the indexes of the similarity matrix
                       If document 421 is at the 5th index (row, column) of the similarity matrix,
                       it should be on the 5th index of list_docs.
            mmr_lambda: The hyperparameter lambda used to measure the MMR scores of each document

        Returns:
            A list containing tuples of the ranked documents and their MMR scores when the documents were added to S
        """
        # NOTE: This algorithm implementation requires some amount of planning as you need to maximize
        #       the MMR at every step.
        #       1. Create an empty list S
        #       2. Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        #       3. Move that element from R and append it to S
        #       4. Repeat 2 & 3 until there are no more remaining elements in R to be processed

        # 1: Create an empty list S
        S = []

        if len(thresholded_search_results) == 0:
            return []
        
        # 2: Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        while thresholded_search_results:
            max_mmr = -float('inf')

            for docid, score in thresholded_search_results:
                max_similarity_score = -float('inf')

                index = list_docs.index(docid)
                
                if not S:
                    max_similarity_score = 0
                    mmr = mmr_lambda * score - (1 - mmr_lambda) * max_similarity_score
                else:
                    for doc in S:
                       max_similarity_score = max(max_similarity_score, similarity_matrix[index][list_docs.index(doc[0])])
                       mmr = mmr_lambda * score - (1 - mmr_lambda) * max_similarity_score

                if mmr > max_mmr:
                    max_mmr = mmr
                    new_added_doc = (docid, score)

            # 3: Move that element from R and append it to S
            S.append((new_added_doc[0], max_mmr))
            thresholded_search_results.remove(new_added_doc)

        return S


    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None, mmr_lambda:int=1, mmr_threshold:int=100) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking

        # TODO: Construct the feature vectors for each query-document pair in the top 100

        # TODO: Use your L2R model to rank these top 100 documents
        
        # TODO: Sort posting_lists based on scores
        
        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked

        # TODO Run MMR diversification for appropriate values of lambda

        # TODO Get the threholded part of the search results, aka top t results and
        #      keep the rest separate

        # TODO Get the document similarity matrix for the thresholded documents using vector_ranker
        #      Preserve the input list of documents to be used in the MMR function
        
        # TODO Run the maximize_mmr function with appropriate arguments
        
        # TODO Add the remaining search results back to the MMR diversification results
        
        # TODO: Return the ranked documents
        
        initial_scores = self.ranker.query(query, pseudofeedback_num_docs=pseudofeedback_num_docs,
                                           pseudofeedback_alpha=pseudofeedback_alpha,
                                           pseudofeedback_beta=pseudofeedback_beta, user_id=user_id)

        query_parts = self.document_preprocessor.tokenize(query)
        if len(query_parts) == 0:
            return []

        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)

        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

        top_100_docs = initial_scores[:100]
        above_100_docs = initial_scores[100:]

        feature_vectors = []
        for doc in top_100_docs:
            feature_vectors.append(self.feature_extractor.generate_features(doc[0],
            doc_word_counts[doc[0]], title_word_counts[doc[0]], query_parts, query=query))

        reranked_scores = []
        reranked_pairs = []
        reranked_scores = self.model.predict(feature_vectors)

        for i in range(len(reranked_scores)):
            docid = top_100_docs[i][0]
            score = reranked_scores[i]
            reranked_pairs.append((docid, score))

        reranked_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_pairs.extend(above_100_docs)

        # return reranked_pairs

        # TODO Run MMR diversification for appropriate values of lambda
        if isinstance(self.ranker, VectorRanker):

        # # TODO Get the threholded part of the search results, aka top t results and
        # #      keep the rest separate
            top_results = reranked_pairs[:mmr_threshold]
            bottom_results = reranked_pairs[mmr_threshold:]

        # # TODO Get the document similarity matrix for the thresholded documents using vector_ranker
        # #      Preserve the input list of documents to be used in the MMR function
            top_results_docids = [docid for docid, score in top_results]
            similarity_matrix = self.ranker.document_similarity(list_docs=top_results_docids)

        # # TODO Run the maximize_mmr function with appropriate arguments
            mmr_results = self.maximize_mmr(top_results, similarity_matrix, top_results_docids, mmr_lambda)

        # # TODO Add the remaining search results back to the MMR diversification results
            mmr_results.extend(bottom_results)

        # # TODO: Return the ranked documents

            return mmr_results
        else:
            return reranked_pairs


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments

        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing

        # TODO Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring

        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features
        self.cross_encoder = ce_scorer

        self.doc_TF_IDF = TF_IDF(document_index)
        self.title_TF_IDF = TF_IDF(title_index)

        self.bm25 = BM25(document_index)
        self.PivotedNormalization = PivotedNormalization(document_index)
        self.YourRanker = YourRanker(document_index)

    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        article_length = self.document_index.get_doc_metadata(docid)
        return article_length['length']

    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        title_length = self.title_index.get_doc_metadata(docid)
        return title_length['length']

    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        tf_score = 0.0

        for query in set(query_parts):
            if query is None:
                continue
            if query not in word_counts:
                continue

            
            term_frequency = word_counts[query]
            if term_frequency == 0:
                continue

            tf_score += math.log(term_frequency + 1)
        
        return tf_score

    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        if index == self.title_index:
            tf_idf = self.title_TF_IDF
        else:
            tf_idf = self.doc_TF_IDF
        score = tf_idf.score(docid=docid, doc_word_counts=word_counts, query_word_counts=Counter(query_parts))
        return score

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        score = self.bm25.score(docid, doc_word_counts, query_word_counts=Counter(query_parts))
        return score

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        score =  self.PivotedNormalization.score(docid, doc_word_counts, Counter(query_parts))
        return score

    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        recognized_categories = list(self.recognized_categories)

        document_categories = self.doc_category_info[docid]
        binary_features = [1 if category in document_categories else 0 for category in recognized_categories]

        return binary_features

    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        try:
            return self.docid_to_network_features[docid]['pagerank']
        except:
            return 0.0

    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        try:
            return self.docid_to_network_features[docid]['hub_score']
        except:
            return 0.0

    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        try:
            return self.docid_to_network_features[docid]['authority_score']
        except:
            0.0

    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        try:
            return self.cross_encoder.score(docid, query)
        except:
            return 0.0

    def get_your_ranker(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates a score using YourRanker.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The score of my ranker 
        """

        score =  self.YourRanker.score(docid, doc_word_counts, Counter(query_parts))
        return score
    

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        document_length = len(doc_word_counts)
        feature_vector.append(document_length)

        # TODO: Title Length
        title_length = len(title_word_counts)
        feature_vector.append(title_length)

        # TODO: Query Length
        query_length = len(query_parts)
        feature_vector.append(query_length)

        # TODO: TF (document)
        tf = self.get_tf(index=self.document_index, docid=docid, word_counts=doc_word_counts, query_parts=query_parts)
        feature_vector.append(tf)

        # TODO: TF-IDF (document)
        tf_idf = self.get_tf_idf(index=self.document_index, docid=docid, word_counts=doc_word_counts, query_parts=query_parts)
        feature_vector.append(tf_idf)

        # TODO: TF (title)
        tf_title = self.get_tf(index=self.title_index, docid=docid, word_counts=title_word_counts, query_parts=query_parts)
        feature_vector.append(tf_title)

        # TODO: TF-IDF (title)
        tf_idf_title = self.get_tf_idf(index=self.title_index, docid=docid, word_counts=title_word_counts, query_parts=query_parts)
        feature_vector.append(tf_idf_title)

        # TODO: BM25
        bm25 = self.get_BM25_score(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts)
        feature_vector.append(bm25)

        # TODO: Pivoted Normalization
        pivoted_norm = self.get_pivoted_normalization_score(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts)
        feature_vector.append(pivoted_norm)

        # TODO: PageRank
        pagerank = self.get_pagerank_score(docid)
        feature_vector.append(pagerank)

        # TODO: HITS Hub
        hub_score = self.get_hits_hub_score(docid)
        feature_vector.append(hub_score)

        # TODO: HITS Authority
        authority_score = self.get_hits_authority_score(docid)
        feature_vector.append(authority_score)

        # TODO: Cross-Encoder Score
        cross_encoder_score = self.get_cross_encoder_score(docid, query)
        feature_vector.append(cross_encoder_score)

        # TODO: Add at least one new feature to be used with your L2R model
        your_ranker_score = self.get_your_ranker(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts)
        feature_vector.append(your_ranker_score)

        # TODO: Document Categories
        #       This should be a list of binary values indicating which categories are present
        categories = self.get_document_categories(docid)
        feature_vector.extend(categories)

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.params = default_params
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generate the predicted values using the LGBMRanker
        featurized_docs_array = np.array(featurized_docs)

        predictions = self.model.predict(featurized_docs_array)

        return predictions

