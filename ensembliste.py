import numpy as np
from preprocessing import vocabulary, preprocess_document
import enum


class Mesure(enum.Enum):
    dice = "dice"
    jaccard = "jaccard"
    sym_norm = "sym_norm"


class Ensembliste:

    def __init__(self, mesure_type=Mesure.dice):
        """
        :param mesure_type: class Mesure
        """
        self.mesure_type = mesure_type
        self.index = None
        self.inverted_index = None
        self.stop_list = np.genfromtxt('data/stoplist/stoplist-english.txt', dtype='str')

    def mesure(self, D, Q):
        """
        Computes similarity between D and Q based on mesure type define in constructor.
        :param D: document (list of unique tokens)
        :param Q: query (list of unique tokens)
        :return: float
        """
        if self.mesure_type == Mesure.dice:
            return self.mesure_dice(D, Q)
        elif self.mesure_type == Mesure.jaccard:
            return self.mesure_jaccard(D, Q)
        else:
            return self.mesure_sym_norm(D, Q)

    @staticmethod
    def descripteur_ensembliste_document(document):
        """
        Creates ensembliste descriptor
        :param document: must be bag words dict {"word" : frequency}
        :return: list of unique word in document
        """
        return [word for word in document.keys()]

    @staticmethod
    def mesure_dice(D, Q):
        num = len(np.intersect1d(D, Q))
        den = len(set(D)) + len(set(Q))
        return 2 * num / den

    @staticmethod
    def mesure_jaccard(D, Q):
        num = len(np.intersect1d(D, Q))
        den = len(list(set(D) | set(Q)))
        return num / den

    def mesure_sym_norm(self, D, Q):
        return 1 - self.mesure_dice(D, Q)

    def index_construction(self, corpus_tokenized):
        """
        Builds linear index.
        :param corpus_tokenized: List of documents (one document is a bag of words)
        :return: dict{"document_id" : descripteur_ensembliste_document(document)}
        """
        result = dict()
        for i, document in enumerate(corpus_tokenized):
            result[i] = self.descripteur_ensembliste_document(document)
        self.index = result
        return result

    def linear_index_search(self, q):
        """
        Searches in linear index.
        :param q:  must be bag words dict {"word" : frequency}
        :return: result -> list of tuples (id of document, measure of similarity) sorted by similarity
        """
        if self.index is None:
            raise Exception("Please build index before searching")
        result = []
        q_bow = self.descripteur_ensembliste_document(q)
        for key in self.index:
            result.append((key, self.mesure(self.index[key], q_bow)))
        result.sort(key=lambda tup: tup[1], reverse=True)
        return result

    def inverted_index_construction(self, corpus_tokenized):
        """
        Builds inverted index.
        :param corpus_tokenized: list of documents, each document must be bag words dict {"word" : frequency}
        :return: inverted index -> dictionary {"descriptors" : lists of lists -> each list is a document
                                                        each sublist contains all unique words in corresponding document
                                                "inverted" : dictionary {"word" : list[ids of documents containing word]
                                                }
        """
        index_inverted = dict()
        index_descriptors = dict()
        vocab = vocabulary(corpus_tokenized)

        for mot in vocab:
            index_inverted[mot] = []

        for d_id, document in enumerate(corpus_tokenized):
            bow = self.descripteur_ensembliste_document(document)
            index_descriptors[d_id] = bow
            for word in bow:
                index_inverted[word].append(d_id)

        index = dict()
        index["descriptors"] = index_descriptors
        index["inverted"] = index_inverted

        self.inverted_index = index

        return index

    def inverted_index_search(self, q):
        """
        Searches in inverted index.
        :param q: must be bag words dict {"word" : frequency}
        :return: result -> list of tuples (id of document, measure of similarity) sorted by similarity
        """

        if self.inverted_index is None:
            raise Exception("Please build inverted index before searching")

        q_bow = self.descripteur_ensembliste_document(q)
        index_descriptors = self.inverted_index["descriptors"]
        index_inverted = self.inverted_index["inverted"]

        short_list = []
        for word in q_bow:
            if word in index_inverted.keys():
                short_list = list(set(index_inverted[word]) | set(short_list))

        results = []
        for d_id in short_list:
            results.append((d_id, self.mesure(index_descriptors[d_id], q_bow)))

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results

    def search_all_queries(self, queries, inverted=False, stop_words=True, stemm=True, bag_words=True):
        """
        Searches a list of queries.
        :param queries: list of queries
        :param inverted: boolean True if using inverted index, False if using linear index
        :param stop_words: boolean True to remove them False otherwise
        :param stemm: boolean True to stemm words False otherwise
        :param bag_words: boolean True to turn documents into bagwords False otherwise
        :return: dict {
                        'retrieved' :list(dict{
                                                'id' : dict{
                                                            'relevant : list(ids (sorted) of similar documents to query)
                                                            }
                                                )}
                        }
        """
        dict_results = dict()
        all_results = list()

        for query in queries:
            result = dict()
            result['id'] = query['id']

            query_bag_words = self.preprocess_query(query['text'], stop_words=stop_words, stemm=stemm,
                                                    bag_words=bag_words)
            result['relevant'] = self.linear_index_search(query_bag_words) \
                if not inverted else self.inverted_index_search(query_bag_words)

            results_id = [res[0] for res in result['relevant']]
            result['relevant'] = results_id
            all_results.append(result)

        dict_results['retrieved'] = all_results
        return dict_results

    def preprocess_query(self, query_text, stop_words=True, stemm=True, bag_words=True):
        """
        Preprocess a query before searching
        :param query_text: str
        :param stop_words: boolean True to remove them False otherwise
        :param stemm: boolean True to stemm words False otherwise
        :param bag_words: boolean True to turn documents into bagwords False otherwise
        :return: result of preprocess_document() in preprocessing.py
        """
        return preprocess_document(query_text, self.stop_list, stop_words=stop_words, stemm=stemm, bag_words=bag_words)
