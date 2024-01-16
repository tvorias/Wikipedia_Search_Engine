# Wikipedia_Search_Engine
Code to manually build a search engine

## SI 650/EECS 549 - Homework
### Use Case
This information retrieval system was developed as homework for the University of Michigan's SI 650/EECS 549 Information Retrieval course. The goal of this course-long project was to manually develop a search engine system from the ground-up to provide a solid understanding of how an information retrieval system works. There are many Python packages (such as PyTerrier) that can do these steps in a few lines of code, such as creating an inverted index. However, this provides a good overview of how those packages and functions work behind the scenes. Because of that, the various rankers may not perform as well in terms of accuracy or efficiency. Refinements can be made to support optimization. This repository aims to help those interested in learning the structure of an information retrieval system. 

Because I do not have the rights to distribute the Wikipedia data or the hand-ranked query-document relevancy data necessary to train the various models, this repo only contains the general files for building the IR system. The system is built to work with that specific data, so changes will need to be made based on the structure of the data used with it. 

## Set Up
### Dependencies
- `Python3.11`
[Installation information can be found here](https://www.python.org/downloads/release/python-3112/)

### Installations
This is a Python 3.11 FastAPI project with the necessary requirements added to the `requirements.txt` file. To install the packages in the file:
-  `python -m pip install -r requirements.txt`

## Running the program
Without the necessary files, this IR system cannot be run. However, to explore the files and make changes based on your own data, you will need download all code files: 
  - `document_preprocessor.py`  This file provides classes and supporting methods related to preprocessing documents, including removing stopwords, handling multi-word expressions, and document augmentation.
  - `indexing.py`  This file provides classes to manually create an inverted index. Additional starter code is included for creating a positional inverted index.
  - `ranker.py`  This file provides classes for implementing rankers for the search engine, including BM25 (and personalized BM25 which incorporates user input), TF-IDF, Pivoted Normalization, and Cross encoder. There is additional starter code for word count cosine similarity and Dirichlet LM rankers.
  - `l2r.py`  This file provides classes for creating a Learning to Rank (L2R) system and a LambdaMART (LGBRanker) model.
  - `relevance.py`  This file provides functions for evaluating model performance, including mean average precision (MAP), normalized discounted cumulative gain (NDCG), and normalized fairness-aware rank retrieval (NFaiRR).
  - `vector_ranker.py` This file provides a class for utilizing a bi-encoder model.

