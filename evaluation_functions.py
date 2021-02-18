import numpy as np
import json
from preprocessing import preprocess_corpus
from src.ir_evaluation import Evaluator
from ensembliste import Ensembliste
from vectoriel import Vectoriel
import matplotlib.pyplot as plt
import time
import pandas as pd


def load_data(dataset_name='med'):
    """
    Loads data from a given dataset name
    !!!Datasets must be json files in /data/datasets/ folder !!!
    :param dataset_name: str
    :return: dataset (corpus), queries unlabelled, queries labels
    """
    path_data_folder = 'data/datasets/'
    dataset_path = path_data_folder + dataset_name + '_dataset.json'
    queries_path = path_data_folder + dataset_name + '_queries.json'
    ground_truth_path = path_data_folder + dataset_name + '_groundtruth.json'
    with open(dataset_path) as json_file:
        dataset = json.load(json_file)

    with open(queries_path) as json_file:
        queries = json.load(json_file)

    with open(ground_truth_path) as json_file:
        ground_truth = json.load(json_file)

    return dataset['dataset'], queries['queries'], ground_truth['groundtruth']


def plot_precision_recall(evaluation_list, labels):
    """
    Plots multiple recall-precision curves on same plot (for comparison)
    :param evaluation_list: list of tuples [(x1, y1), (x2, y2), ...]
    :param labels: curves title
    :return: None
    """
    for evaluation, label in zip(evaluation_list, labels):
        plt.plot([e[0] for e in evaluation], [e[1] for e in evaluation], label=label)
    plt.grid()
    plt.legend(loc='upper right')


def remove_queries_without_ground_truth(queries, ground_truth):
    new_queries = list()
    for gt in ground_truth:
        for q in queries:
            if gt['id'] == q['id']:
                new_queries.append(q)
                break
    return new_queries


def eval_ensembliste(dataset_name='med', stop_words=True, stemm=True, bag_words=True):
    """
    Evaluates performances of ensembliste SRI with a given dataset name
    Boolean parameters allows to see their impacts in the SRI performances
    :param dataset_name: str
    :param stop_words: boolean
    :param stemm: boolean
    :param bag_words: boolean
    :return: dictionary 'result'
    """
    dataset, queries, ground_truth = load_data(dataset_name)
    queries = remove_queries_without_ground_truth(queries, ground_truth)
    stop_list = np.genfromtxt('data/stoplist/stoplist-english.txt', dtype='str')
    dataset_bagwords = preprocess_corpus(dataset, stop_list, stop_words=stop_words, stemm=stemm, bag_words=bag_words)

    ground_truth_dict = dict()
    ground_truth_dict['groundtruth'] = ground_truth

    ensembliste = Ensembliste()
    result = dict()

    # Linear index
    # Compute time to build linear index
    start_linear_build = time.time()
    index_linear = ensembliste.index_construction(dataset_bagwords)
    end_linear_build = time.time()
    result["linear_build_time"] = end_linear_build - start_linear_build

    # Compute time to search in linear index
    start_search_linear = time.time()
    eval_all_queries = ensembliste.search_all_queries(queries, inverted=False, stop_words=stop_words, stemm=stemm,
                                                      bag_words=bag_words)
    end_search_linear = time.time()
    result["linear_search_time"] = end_search_linear - start_search_linear

    # Evaluate performances (recall precision)
    eval_ensembliste_linear = Evaluator(retrieved=eval_all_queries, relevant=ground_truth_dict)
    result["evaluation_linear"] = eval_ensembliste_linear.evaluate_pr_points()
    result["MAP_linear"] = eval_ensembliste_linear.evaluate_map()

    # Inverted index
    # Compute time to build inverted index
    start_inverted_build = time.time()
    index_inverted = ensembliste.inverted_index_construction(dataset_bagwords)
    stop_inverted_build = time.time()
    result['inverted_build_time'] = stop_inverted_build - start_inverted_build

    # Compute time to search in inverted index
    start_search_inverted = time.time()
    eval_inverted_all_queries = ensembliste.search_all_queries(queries, inverted=True, stop_words=stop_words,
                                                               stemm=stemm, bag_words=bag_words)
    end_search_inverted = time.time()
    result['inverted_search_time'] = end_search_inverted - start_search_inverted

    # Evaluate performances (recall precision)
    eval_ensembliste_inverted = Evaluator(retrieved=eval_inverted_all_queries, relevant=ground_truth_dict)
    result["evaluation_inverted"] = eval_ensembliste_inverted.evaluate_pr_points()
    result["MAP_inverted"] = eval_ensembliste_inverted.evaluate_map()

    return result


def evaluate_all_datasets(datasets, stop_words=True, stemm=True, bag_words=True):
    """
    Allows to evaluate performances for multiple datasets (for comparison)
    :param datasets: str
    :param stop_words: boolean
    :param stemm: boolean
    :param bag_words: boolean
    :return: dictionary of dictionaries (result of eval_ensembliste()), keys are dataset names
    """
    results = dict()
    for dataset in datasets:
        print(f'[LOG] evaluating {dataset} performances...')
        results[dataset] = eval_ensembliste(dataset_name=dataset, stop_words=stop_words, stemm=stemm,
                                            bag_words=bag_words)
    return results


def plot_performances(results, subplot_value, title="Recall-precision curves with ensembliste model", save_plot=False,
                      model='Ensembliste', figsize=(15, 15)):
    """
    Plots performances with subplots. Each subplot compare linear and inverted index from a given datasets name
    :param figsize: size of figure
    :param model: str for curve label in plot
    :param results: dictionary (result of evaluate_all_datasets)
    :param subplot_value: location of current subplot
    :param title: str
    :param save_plot: boolean
    :return: None
    """
    plt.figure(figsize=figsize)
    subplot = subplot_value
    for dataset_name in results:
        result = results[dataset_name]
        subplot += 1

        eval_linear = result["evaluation_linear"]
        label_linear = f"{model} linear index {dataset_name} dataset"
        eval_inverted = result["evaluation_inverted"]
        label_inverted = f"{model} inverted index {dataset_name} dataset"
        plt.subplot(subplot)

        plot_precision_recall([eval_linear, eval_inverted], [label_linear, label_inverted])
    plt.suptitle(title, fontsize=20)
    if save_plot:
        plt.savefig(f'{title}.png', transparent=False)
    plt.show()


def merge_results(results_origin, results_custom):
    merged_results = dict()
    for dataset in results_origin:
        res = dict()
        linear = dict()
        inverted = dict()

        linear['origin'] = results_origin[dataset]['evaluation_linear']
        linear['custom'] = results_custom[dataset]['evaluation_linear']

        inverted['origin'] = results_origin[dataset]['evaluation_inverted']
        inverted['custom'] = results_custom[dataset]['evaluation_inverted']

        res['evaluation_linear'] = linear
        res['evaluation_inverted'] = inverted
        merged_results[dataset] = res

    return merged_results


def plot_performances_comparison(merged_results, subplot_value, title="Recall-precision curves with ensembliste model",
                                 origin='origin', custom='custom', save_plot=False):
    """
    Plots performances with subplots. Each subplot compare linear and inverted index from a given datasets name
    :param custom: str for plot legend
    :param origin: str for plot legend
    :param merged_results: dictionary (result of merge_results())
    :param subplot_value: location of current subplot
    :param title: str
    :param save_plot: boolean
    :return: None
    """
    plt.figure(figsize=(15, 15))
    subplot = subplot_value
    for dataset_name in merged_results:
        result = merged_results[dataset_name]
        subplot += 1

        eval_linear_origin = result["evaluation_linear"]['origin']
        label_linear_origin = f"Linear {origin} index {dataset_name} dataset"

        eval_linear_custom = result["evaluation_linear"]['custom']
        label_linear_custom = f"Linear {custom} index {dataset_name} dataset"

        eval_inverted_origin = result["evaluation_inverted"]['origin']
        label_inverted_origin = f" Inverted {origin} index {dataset_name} dataset"

        eval_inverted_custom = result["evaluation_inverted"]['custom']
        label_inverted_custom = f" Inverted {custom} index {dataset_name} dataset"

        plt.subplot(subplot)

        plot_precision_recall([eval_linear_origin, eval_inverted_origin, eval_linear_custom, eval_inverted_custom],
                              [label_linear_origin, label_inverted_origin, label_linear_custom, label_inverted_custom])
    plt.suptitle(title, fontsize=20)
    if save_plot:
        plt.savefig(f'{title}.png', transparent=False)
    plt.show()


def compare_time(datasets_name, all_results):
    """

    :param datasets_name: list of str
    :param all_results: dictionary like this
    all_results = {"results_fully_preprocessed": results_fully_preprocessed,
               "results_stopwords": results_stopwords,
               "results_stemmer" :results_stemmer,
               "results_raw_data" :results_raw_data}
    :return: dictionary with Dataframe to compare time easily results for all datasets
    """
    datasets_times = dict()
    for name in datasets_name:
        dict_compare = dict()
        for res in all_results:
            times = dict()
            times["index_build_linear"] = all_results[res][name]['linear_build_time']
            times["index_build_inverted"] = all_results[res][name]['inverted_build_time']
            times["index_build_comp_slower"] = all_results[res][name]['inverted_build_time'] / all_results[res][name][
                'linear_build_time']
            times["queries_search_linear"] = all_results[res][name]['linear_search_time']
            times["queries_search_inverted"] = all_results[res][name]['inverted_search_time']
            times["queries_search_comp_faster"] = all_results[res][name]['linear_search_time'] / all_results[res][name][
                'inverted_search_time']
            dict_compare[res] = times
        datasets_times[name] = pd.DataFrame.from_dict(dict_compare, orient='index')
    return datasets_times


def evaluate_all_datasets_tfidf(datasets, tf_idf=True, stop_words=True, stemm=True, bag_words=True):
    """
    Allows to evaluate performances for multiple datasets (for comparison)
    :param tf_idf: boolean
    :param datasets: str
    :param stop_words: boolean
    :param stemm: boolean
    :param bag_words: boolean
    :return: dictionary of dictionaries (result of eval_ensembliste()), keys are dataset names
    """
    results = dict()
    for dataset in datasets:
        print(f'[LOG] evaluating {dataset} performances...')
        results[dataset] = eval_tfidf(dataset_name=dataset, tf_idf=tf_idf, stop_words=stop_words, stemm=stemm,
                                      bag_words=bag_words)
    return results


def eval_tfidf(dataset_name='med', tf_idf=True, stop_words=True, stemm=True, bag_words=True):
    """
    Evaluates performances of ensembliste SRI with a given dataset name
    Boolean parameters allows to see their impacts in the SRI performances
    :param tf_idf: boolean
    :param dataset_name: str
    :param stop_words: boolean
    :param stemm: boolean
    :param bag_words: boolean
    :return: dictionary 'result'
    """
    dataset, queries, ground_truth = load_data(dataset_name)
    stop_list = np.genfromtxt('data/stoplist/stoplist-english.txt', dtype='str')
    dataset_bagwords = preprocess_corpus(dataset, stop_list, stop_words=stop_words, stemm=stemm, bag_words=bag_words)

    ground_truth_dict = dict()
    ground_truth_dict['groundtruth'] = ground_truth

    vectoriel = Vectoriel(dataset_bagwords)
    result = dict()

    # Linear index
    # Compute time to build linear index
    start_linear_build = time.time()
    index_linear = vectoriel.index_construction(dataset_bagwords, tf_idf=tf_idf)
    end_linear_build = time.time()
    result["linear_build_time"] = end_linear_build - start_linear_build

    # Compute time to search in linear index
    start_search_linear = time.time()
    eval_all_queries = vectoriel.search_all_queries(queries, inverted=False, tf_idf=tf_idf, stop_words=stop_words,
                                                    stemm=stemm, bag_words=bag_words)
    end_search_linear = time.time()
    result["linear_search_time"] = end_search_linear - start_search_linear

    # Evaluate performances (recall precision)
    eval_tfidf_linear = Evaluator(retrieved=eval_all_queries, relevant=ground_truth_dict)
    result["evaluation_linear"] = eval_tfidf_linear.evaluate_pr_points()
    result["MAP_linear"] = eval_tfidf_linear.evaluate_map()

    # Inverted index
    # Compute time to build inverted index
    start_inverted_build = time.time()
    index_inverted = vectoriel.inverted_index_construction(dataset_bagwords, tf_idf=tf_idf)
    stop_inverted_build = time.time()
    result['inverted_build_time'] = stop_inverted_build - start_inverted_build

    # Compute time to search in inverted index
    start_search_inverted = time.time()
    eval_inverted_all_queries = vectoriel.search_all_queries(queries, inverted=True, tf_idf=tf_idf,
                                                             stop_words=stop_words, stemm=stemm, bag_words=bag_words)
    end_search_inverted = time.time()
    result['inverted_search_time'] = end_search_inverted - start_search_inverted

    # Evaluate performances (recall precision)
    eval_tfidf_inverted = Evaluator(retrieved=eval_inverted_all_queries, relevant=ground_truth_dict)
    result["evaluation_inverted"] = eval_tfidf_inverted.evaluate_pr_points()
    result["MAP_inverted"] = eval_tfidf_inverted.evaluate_map()

    return result
