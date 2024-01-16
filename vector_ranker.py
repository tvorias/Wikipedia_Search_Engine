from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np
from collections import Counter

class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        self.row_to_docid = row_to_docid
        self.bi_encoder_model = SentenceTransformer(bi_encoder_model_name, device='cpu')
        self.document_embeddings = encoded_docs

    def query(self, query: str, pseudofeedback_num_docs=0,
              pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: We don't use the user_id parameter in vector ranker. It is here just to align all the
                    Ranker interfaces.

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """

        # NOTE: Do not forget to handle edge cases on the input

        # TODO: Encode the query using the bi-encoder
        query_emb = self.bi_encoder_model.encode(query, convert_to_tensor=True)

        # TODO: Score the similarity of the query vec and document vectors for relevance
        scores = util.dot_score(query_emb, self.document_embeddings).cpu().numpy().flatten()
        # print(scores)

        # TODO: Generate the ordered list of (document id, score) tuples
        doc_score_pairs = list(zip(self.row_to_docid, scores))
       
        # TODO: Sort the list by relevance score in descending order (most relevant first)

        sorted_doc_store_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)


        # TODO If the user has indicated we should use feedback, then update the
        #  query vector with respect to the specified number of most-relevant documents

            # TODO Get the most-relevant document vectors for the initial query

            # TODO Compute the average vector of the specified number of most-relevant docs
            #  according to how many are to be used for pseudofeedback

            # TODO Combine the original query doc with the feedback doc to use
            #  as the new query embedding

        # TODO: Score the similarity of the query vec and document vectors for relevance

        # TODO: Generate the ordered list of (document id, score) tuples

        # TODO: Sort the list by relevance score in descending order (most relevant first)

    # TODO (HW5): Find the dot product (unnormalized cosine similarity) for the list of documents (pairwise)
    # NOTE: You should return a matrix where element [i][j] would represent similarity between
    #   list_docs[i] and list_docs[j]
    
        if pseudofeedback_num_docs > 0:
            
            relevant_indices = [self.row_to_docid.index(docid) for docid, score in sorted_doc_store_pairs[:pseudofeedback_num_docs]]

            relevant_doc_vectors = self.document_embeddings[relevant_indices]

            avg_feedback_vector = np.mean(relevant_doc_vectors, axis=0)

            query_emb = pseudofeedback_alpha * query_emb + (pseudofeedback_beta) * avg_feedback_vector

            # TODO: Score the similarity of the query vec and document vectors for relevance
            scores = util.dot_score(query_emb, self.document_embeddings).cpu().numpy().flatten()

            # TODO: Generate the ordered list of (document id, score) tuples
            doc_score_pairs = list(zip(self.row_to_docid, scores))
        
            # TODO: Sort the list by relevance score in descending order (most relevant first)

            sorted_doc_store_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return sorted_doc_store_pairs

    def document_similarity(self, list_docs: list[int]) -> np.ndarray:
        """
        Calculates the pairwise similarities for a given list of documents

        Args:
            list_docs: A list of document IDs

        Returns:
            A matrix where element [i][j] is a similarity score between list_docs[i] and list_docs[j]
        """
        
        # TODO (HW5): Find the dot product (unnormalized cosine similarity) for the list of documents (pairwise)
        # NOTE: You should return a matrix where element [i][j] would represent similarity between
        #   list_docs[i] and list_docs[j]

        # edge case if list_docs is empty. Return empty list
        if list_docs is None or len(list_docs) == 0:
            return []
        
        # num_docs = len(list_docs)
        # similarity_matrix = np.zeros((num_docs, num_docs))
        
        # for i in range(num_docs):
        #     for j in range(num_docs):
        #         similarity_matrix[i][j] = np.dot(self.document_embeddings[i], self.document_embeddings[j])

        # doc_embeddings = self.document_embeddings[list_docs]

        # similarity_matrix = np.dot(doc_embeddings, doc_embeddings.T)

        docid_to_row = {docid: row for row, docid in enumerate(self.row_to_docid)}

        filtered_rows = [docid_to_row[docid] for docid in list_docs]
        doc_embeddings = self.document_embeddings[filtered_rows]

        similarity_matrix = np.dot(doc_embeddings, doc_embeddings.T)

        
        return similarity_matrix
