from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import shelve
from document_preprocessor import Tokenizer
import gzip


class IndexType(Enum):
    # The three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    # NOTE: You don't need to support the following three
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = defaultdict(Counter)  # Central statistics of the index
        self.index = {}  # Index
        self.document_metadata = {}  # Metadata like length, number of unique tokens of the documents
        self.vocabulary = set()
        self.term_metadata = {}

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        for term in self.index:
            if docid in self.index[term]:
                del self.index[term][docid]
        if docid in self.document_metadata:
            del self.document_metadata[docid]

        
        terms_to_remove = []
        for term, postings in self.index.items():
            if postings == []:
                terms_to_remove.append(term)
                del self.index[term]

        for term in terms_to_remove:
            if term in self.vocabulary:
                self.vocabulary.discard(term)

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        length = len(tokens)
        if length == 0:
            self.document_metadata[docid] = {'length': 0, 'unique_tokens': 0}
            return
        uniq_toke_ct = 0
        counter = Counter(tokens)
        for token, frequency in counter.items():
            if token:
                self.vocabulary.add(token)
                v = self.index.get(token, [])
                v.append((docid, frequency))
                self.index[token] = v
                uniq_toke_ct += frequency

        self.document_metadata[docid] = {
            'length': len(tokens),
            'unique_tokens': len(set(tokens)), 
            'uniq_token_ct':uniq_toke_ct
        }

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        if doc_id in self.document_metadata:
            return self.document_metadata[doc_id]
        else:
            return {}

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        if term in self.index:
            self.term_metadata['count'] = len(self.index[term])
            return self.term_metadata
        else:
            return None

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic
        total_tokens = sum(meta["length"] for meta in self.document_metadata.values())
        num_doc = len(self.document_metadata)
        mean_doc = total_tokens / num_doc if num_doc > 0 else 0

        stats = {
            'unique_token_count':len(self.vocabulary),
            "total_token_count": total_tokens,
            "number_of_documents": num_doc,
            "mean_document_length": mean_doc
        }
        self.statistics = stats
        return self.statistics

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # TODO: Save the index files to disk
        if not os.path.exists(index_directory_name):
            os.mkdir(index_directory_name)

        directory_path = index_directory_name
        file = index_directory_name + '.json'

        dict_to_save = {'index': self.index, 'doc_metadata': self.document_metadata,
                      'statistics': self.statistics, 'vocab': list(self.vocabulary)}
        
        with open(os.path.join(directory_path, file), 'w') as file:
            file.write(json.dumps(dict_to_save))

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        directory_path = index_directory_name
        file = index_directory_name + '.json'
        if os.path.exists(directory_path):
            with open(os.path.join(directory_path, file), 'r') as file:
                dict_to_open = json.loads(file.read())
                self.index = dict_to_open['index']
                doc = dict_to_open['doc_metadata']
                self.document_metadata = {int(key): value for key, value in doc.items()}
                
                self.statistics = dict_to_open['statistics']
                self.vocabulary = set(dict_to_open['vocab'])


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #    self.statistics['docmap'] = {}
        #    self.index = defaultdict(list)
        #    self.doc_id = 0
  
    # TODO: Implement all the functions mentioned in the interface
    # This is the typical inverted index where each term keeps track of documents and the term count per document
    def remove_doc(self, docid: int) -> None:
        # TODO implement this to remove a document from the entire index and statistics
        
        for term in self.index:
            if docid in self.index[term]:
                del self.index[term][docid]
        if docid in self.document_metadata:
            del self.document_metadata[docid]

        
        terms_to_remove = []
        for term, postings in self.index.items():
            if postings == []:
                terms_to_remove.append(term)
                del self.index[term]

        for term in terms_to_remove:
            if term in self.vocabulary:
                self.vocabulary.discard(term)
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        # TODO implement this to add documents to the index

        uniq_toke_ct = 0
        counter = Counter(tokens)
        for token, frequency in counter.items():
            if token:
                self.vocabulary.add(token)
                v = self.index.get(token, [])
                v.append((docid, frequency))
                self.index[token] = v
                uniq_toke_ct += frequency
        self.document_metadata[docid] = {
            'length': len(tokens),
            'unique_tokens': len(set(tokens)), 
            'uniq_token_ct':uniq_toke_ct
        }

    def get_postings(self, term: str) -> list[tuple[int, str]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        # TODO implement this to fetch a term's postings from the index
        return self.index.get(term, [])
    
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        # TODO implement to fetch a particular documents stored metadata

        if doc_id in self.document_metadata:
            return self.document_metadata[doc_id]
        
        else:
            return {}
        
    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        # TODO implement to fetch a particular terms stored metadata
        if term in self.index:
            self.term_metadata['count'] = len(self.index[term])
            return self.term_metadata
        else:
            return None
        
    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        # TODO calculate statistics like 'unique_token_count', 'total_token_count', 
        #  'number_of_documents', 'mean_document_length' and any other relevant central statistic.
        total_tokens = sum(meta["length"] for meta in self.document_metadata.values())
        num_doc = len(self.document_metadata)
        mean_doc = total_tokens / num_doc if num_doc > 0 else 0

        stats = {
            #"unique_token_count": len([token for token in self.vocabulary if token is not None]),
            'unique_token_count':len(self.vocabulary),
            "total_token_count": total_tokens,
            #'stored_total_token_count': len(self.vocabulary),
            "number_of_documents": num_doc,
            "mean_document_length": mean_doc
        }
        self.statistics = stats
        return self.statistics
    
    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        # TODO save the index files to disk
        

        if not os.path.exists(index_directory_name):
            os.mkdir(index_directory_name)

        directory_path = index_directory_name
        file = index_directory_name + '.json'

        dict_to_save = {'index': self.index, 'doc_metadata': self.document_metadata,
                      'statistics': self.statistics, 'vocab': list(self.vocabulary)}
        
        with open(os.path.join(directory_path, file), 'w') as file:
            file.write(json.dumps(dict_to_save))

    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        # TODO load the index files from disk to a Python object
        directory_path = index_directory_name
        file = index_directory_name + '.json'
        if os.path.exists(directory_path):
            with open(os.path.join(directory_path, file), 'r') as file:
                dict_to_open = json.loads(file.read())
                self.index = dict_to_open['index']
                doc = dict_to_open['doc_metadata']
                self.document_metadata = {int(key): value for key, value in doc.items()}
                self.statistics = dict_to_open['statistics']
                self.vocabulary = set(dict_to_open['vocab'])


## If interested in using a positional inverted index, this is starter code that can be used. The code in written_responses.ipynb
## is not set up to support this index type, so you will need to modify it to support this index type.
                
class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self, index_name) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #   self.statistics['offset'] = [0]
        #   self.statistics['docmap'] = {}
        #   self.doc_id = 0
        #   self.postings_id = -1




class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        """
        Creates an inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text
                and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index
        """
         
        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
                      
        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency


        # check if index_type is valid
        if index_type == 'PositionalIndex':
            index = PositionalInvertedIndex()
        elif index_type == 'InvertedIndex':
            index = BasicInvertedIndex()
        else:
            index = BasicInvertedIndex()

        # check if stopwords is valid
        if stopwords is None:
            stopwords = set()

        
        doc_list = []

        if dataset_path.endswith('.gz'):
            file = gzip.open(dataset_path, 'rt')
            for line in file:
                doc_list.append(json.loads(line))
        else:
            with open(dataset_path) as file:
                for line in file:
                    doc_list.append(json.loads(line))

        word_frequencies = Counter()
        not_to_index = []

        
        for doc in tqdm(doc_list):
            
            tokens = document_preprocessor.tokenize(doc[text_key])
            if doc_augment_dict:
                if doc['docid'] in doc_augment_dict:
                    for term in doc_augment_dict[doc['docid']]:
                        tokenized_term = document_preprocessor.tokenize(term)
                        tokens.extend(tokenized_term)

            word_frequencies.update(tokens)

        if minimum_word_frequency > 1:
            not_to_index = [word for word in word_frequencies if word_frequencies[word] < minimum_word_frequency]

        if len(stopwords) > 0:
            not_to_index = stopwords.union(set(not_to_index))

        for doc in tqdm(doc_list):
            if max_docs == 0:
                break
            
            tokens = document_preprocessor.tokenize(doc[text_key])
            if doc_augment_dict:
                if doc['docid'] in doc_augment_dict:
                    for term in doc_augment_dict[doc['docid']]:
                        tokenized_term = document_preprocessor.tokenize(term)
                        tokens.extend(tokenized_term)

            tokenized = [word if word not in not_to_index else None for word in tokens]

            index.add_doc(doc['docid'], tokenized)

            max_docs -= 1

        return index

