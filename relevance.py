import math
import csv
from tqdm import tqdm
import pandas as pd


# TODO Implement NFaiRR
def nfairr_score(actual_omega_values: list[int], cut_off=200) -> float:
    """
    Computes the normalized fairness-aware rank retrieval (NFaiRR) score for a list of omega values
    for the list of ranked documents.
    If all documents are from the protected class, then the NFaiRR score is 0.

    Args:
        actual_omega_values: The omega value for a ranked list of documents
            The most relevant document is the first item in the list.
        cut_off: The rank cut-off to use for calculating NFaiRR
            Omega values in the list after this cut-off position are not used. The default is 200.

    Returns:
        The NFaiRR score
    """
    
    nfairr_score = 0.0

    if len(actual_omega_values) < cut_off:
        cut_off = len(actual_omega_values)
    
    # Check if all documents are from the protected class
    if sum(actual_omega_values) == 0:
        return 0.0
    
    actual_omega_values = actual_omega_values[:cut_off]
    # print(actual_omega_values)

    sorted_values = sorted(actual_omega_values, reverse=True)


    ideal_fairness = sorted_values[0]
    for i in range(1, cut_off):
        ideal_fairness += sorted_values[i] * (1 / (math.log2(i + 1)))

    fairness_score = actual_omega_values[0]
    for i in range(1, cut_off):
        fairness_score += actual_omega_values[i] * (1 / (math.log2(i + 1)))

    nfairr_score = fairness_score / ideal_fairness if ideal_fairness > 0 else 0.0

    return nfairr_score


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """

    if cut_off <= 0:
        raise ValueError("cut_off must be greater than 0")

    relevant_count = 0
    precision_at_k_sum = 0.0
    
    for i, result in enumerate(search_result_relevances):
        if i >= cut_off:
            break
        if result > 0:
            relevant_count += 1
            precision_at_k = relevant_count / (i + 1)
            precision_at_k_sum += precision_at_k

    if relevant_count == 0:
        return 0.0

    average_precision = precision_at_k_sum / cut_off

    return average_precision


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    
    if cut_off <= 0:
        raise ValueError("cut_off must be greater than 0")

    search_result_relevances = search_result_relevances[:cut_off]
    ideal_relevance_score_ordering = ideal_relevance_score_ordering[:cut_off]


    dcg = search_result_relevances[0]
    for i in range(1, len(search_result_relevances)):
        dcg += search_result_relevances[i] / math.log2(i + 1) 

    
    idcg = ideal_relevance_score_ordering[0]
    for i in range(1, len(ideal_relevance_score_ordering)):
        idcg += ideal_relevance_score_ordering[i] / math.log2(i + 1)


    if idcg == 0:
        return 0.0
    ndcg = dcg / idcg

    return ndcg


def run_relevance_tests(relevance_data_filename: str, ranker, if_list=True, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None, mmr_lambda=1) -> dict[str, float]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.
        if_list: If true, return a list of scores for each query. If false, return the average of all scores.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    relevance = []
    with open(relevance_data_filename) as r:
        relevance_data = csv.DictReader(r)
        for data in relevance_data:
            relevance.append(data)

    queries = set ()
    for row in relevance:
        queries.add(row['query'])

    map_sum = 0
    ndcg_sum = 0
    map_scores = []
    ndcg_scores = []

    for query in queries:
        results = ranker.query(query, pseudofeedback_num_docs, pseudofeedback_alpha, pseudofeedback_beta, user_id, mmr_lambda)
        ideal_results = []
        for row in relevance:
            if row['query'] == query:
                ideal_results.append((int(row['rel']), int(row['docid'])))

        map_actual_results = []
        for result in results:
            found = False
            for i in range(len(ideal_results)):
                if ideal_results[i][1] == result[0]:
                    if ideal_results[i][0] > 3:
                        map_actual_results.append(1)
                    else:
                        map_actual_results.append(0)
                    found = True
            if not found:
                map_actual_results.append(0)

        ndcg_actual_results = []
        for result in results:
            found = False
            for i in range (len(ideal_results)):
                if ideal_results[i][1] == result[0]:
                    ndcg_actual_results.append(ideal_results[i][0])
                    found = True
            if not found:
                ndcg_actual_results.append(0)

        map_res = map_score(map_actual_results)
        ndcg_res = ndcg_score(ndcg_actual_results, [x[0] for x in ideal_results])
        map_sum += map_res
        ndcg_sum += ndcg_res
        map_scores.append(map_res)
        ndcg_scores.append(ndcg_res)
    if if_list:
        return (map_scores, ndcg_scores)
    else:
        mean_map = map_sum / len(queries)
        mean_ndcg = ndcg_sum / len(queries)
    
        return {'map': mean_map, 'ndcg': mean_ndcg}


# TODO Implement NFaiRR metric for a list of queries to measure fairness for those queries
# NOTE: This has no relation to relevance scores and measures fairness of representation of classes
def run_fairness_test(attributes_file_path: str, protected_class: str, queries: list[str],
                      ranker, cut_off: int = 200) -> float:
    """
    Measures the fairness of the IR system using the NFaiRR metric.

    Args:
        attributes_file_path: The filename containing the documents about people and their demographic attributes
        protected_class: A specific protected class (e.g., Ethnicity, Gender)
        queries: A list containing queries
        ranker: A ranker configured with a particular scoring function to search through the document collection
        cut_off: The rank cut-off to use for calculating NFaiRR

    Returns:
        The average NFaiRR score across all queries
    """
    # TODO Load person-attributes.csv
 
    # TODO Find the documents associated with the protected class

    # TODO Loop through the queries and
    #       1. Create the list of omega values for the ranked list.
    #       2. Compute the NFaiRR score
    # NOTE: This fairness metric has some 'issues' (and the assignment spec asks you to think about it)

    score = []

    for query in queries:
        # Results format = [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        results = ranker.query(query)
        omega_values = []
        for result in results:
            with open(attributes_file_path) as f:
                attributes = csv.DictReader(f)
                for row in attributes:
                    if row['docid'] == result[0]:
                        if row[protected_class] == '1':
                            omega_values.append(1)
                        else:
                            omega_values.append(0)
        score.append(nfairr_score(omega_values, cut_off))
    avg_score = sum(score) / len(score)
    return avg_score

