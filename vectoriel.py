import numpy as np
from preprocessing import vocabulary, preprocess_document


class Vectoriel:

    def __init__(self, corpus_tokenized):
        """
        :param corpus_tokenized: list of documents tokenized
        """
        self.vocabulary = vocabulary(corpus_tokenized)
        self.corpus_tokenized = corpus_tokenized
        self.index = None
        self.inverted_index = None
        self.stop_list = np.genfromtxt('data/stoplist/stoplist-english.txt', dtype='str')

    @staticmethod
    def mesure(v1, v2):
        """
        Computes cosine similarity
        :param v1: vector 1D
        :param v2: vector 1D
        :return: float
        vectors v1 v2 must have the same shape
        """
        return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def compute_idf(self, word):
        """
        Computes idf for a given word
        :param word: token
        :return: log(len(corpus)/number_of_documents_containing_word)
        Word must be in corpus
        """
        if word not in self.vocabulary:
            raise Exception(f"Impossible to compute IDF because '{word}', not in corpus vocab")
        df_t = 0
        for document in self.corpus_tokenized:
            if word in document:
                df_t += 1
        return np.log(len(self.corpus_tokenized) / df_t)

    def descripteur_ensembliste_document(self, document_bagwords, tf_idf=True):
        """
        Creates tf-idf descriptor
        :param document_bagwords: must be bag words dict {"word" : frequency}
        :param tf_idf: boolean True to compute idf else False
        :return: dict{"token" : idf * frequency} if tf_idf True else dict{"token" : frequency}
        """
        result = dict()
        # init all words to 0
        for word in self.vocabulary:
            result[word] = 0
        # add frequency
        for word in document_bagwords:
            if word in self.vocabulary:
                # this if statement is made to avoid to add unknown words from queries
                tf = document_bagwords[word]
                if not tf_idf:
                    idf = 1
                else:  # compute idf
                    idf = self.compute_idf(word)
                result[word] = tf * idf

        return result

    def index_construction(self, corpus_tokenized, tf_idf=True):
        """
        Builds linear index.
        :param corpus_tokenized: List of documents (one document is a bag of words)
        :param tf_idf: tf_idf: boolean True to compute idf else False
        :return: dict{"document_id" : descripteur_ensembliste_document(document)}
        """
        result = dict()
        for i, document in enumerate(corpus_tokenized):
            result[i] = self.descripteur_ensembliste_document(document, tf_idf)
        self.index = result
        return result

    def linear_index_search(self, q, tf_idf=True):
        """
        Searches in linear index.
        :param tf_idf: boolean default True
        :param q:  must be bag words dict {"word" : frequency}
        :return: result -> list of tuples (id of document, measure of similarity)
        """
        if self.index is None:
            raise Exception("Please build index before searching")
        result = []
        q_bow = self.descripteur_ensembliste_document(q, tf_idf)
        for key in self.index:
            vector_document = np.array([e for e in self.index[key].values()])
            vector_q_bow = np.array([e for e in q_bow.values()])
            result.append((key, self.mesure(vector_document, vector_q_bow)))
        result.sort(key=lambda tup: tup[1], reverse=True)
        return result

    def inverted_index_construction(self, corpus_tokenized, tf_idf=True):
        """
        Builds inverse index.
        :param tf_idf: boolean default True
        :param corpus_tokenized: list of documents, each document must be bag words dict {"word" : frequency}
        :return: inverted index -> dictionary {"descriptors" : lists of lists -> each list is a document
                                                        each sublist contains all unique words in corresponding document
                                                "inverted" : dictionary {"word" : list[ids of documents containing word]
                                                }
        """
        index_inverted = dict()
        index_descriptors = dict()

        for mot in self.vocabulary:
            index_inverted[mot] = []

        for d_id, document in enumerate(corpus_tokenized):
            bow = self.descripteur_ensembliste_document(document, tf_idf)
            index_descriptors[d_id] = bow
            for word in bow:
                if bow[word] != 0:
                    index_inverted[word].append(d_id)

        index = dict()
        index["descriptors"] = index_descriptors
        index["inverted"] = index_inverted

        self.inverted_index = index

        return index

    def inverted_index_search(self, q, tf_idf=True):
        """
        Searches in inverted index.
        :param tf_idf: boolean True to compute idf else False
        :param q: must be bag words dict {"word" : frequency}
        :return: result -> list of tuples (id of document, measure of similarity) sorted by similarity
        """

        if self.inverted_index is None:
            raise Exception("Please build inverted index before searching")

        q_bow = self.descripteur_ensembliste_document(q, tf_idf)
        index_descriptors = self.inverted_index["descriptors"]
        index_inverted = self.inverted_index["inverted"]

        short_list = []
        for word in q_bow:
            if word in index_inverted.keys() and q_bow[word] > 0:
                short_list = list(set(index_inverted[word]) | set(short_list))

        results = []
        for d_id in short_list:
            vector_document = np.array([e for e in index_descriptors[d_id].values()])
            vector_q_bow = np.array([e for e in q_bow.values()])
            results.append((d_id, self.mesure(vector_document, vector_q_bow)))

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results

    def search_all_queries(self, queries, inverted=False, tf_idf=True, stop_words=True, stemm=True, bag_words=True):
        """
        Searches a list of queries.
        :param tf_idf: boolean True to compute idf else False
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
            if inverted:
                result['relevant'] = self.inverted_index_search(query_bag_words, tf_idf=tf_idf)
            else:
                result['relevant'] = self.linear_index_search(query_bag_words, tf_idf=tf_idf)

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
