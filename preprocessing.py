from src.tokenizer import Tokenizer, REGEX_SPECIAL_CHARACTERS
from nltk.stem import SnowballStemmer


def preprocess_corpus(corpus, stop_list, stop_words=True, stemm=True, bag_words=True):
    """
    Preprocess all documents in a corpus.
    :param corpus: list of documents (a document is a dict {"text" : "content of document"})
    :param stop_list: array of stop words to remove from documents
    :param stop_words: boolean True to remove them False otherwise
    :param stemm: boolean True to stemm words False otherwise
    :param bag_words: boolean True to turn documents into bagwords False otherwise
    :return: list of documents preprocessed
    """
    corpus_preprocessed = list()
    for element in corpus:
        document = preprocess_document(element['text'], stop_list, stop_words, stemm, bag_words)
        corpus_preprocessed.append(document)

    return corpus_preprocessed


def preprocess_document(document, stop_list, stop_words=True, stemm=True, bag_words=True):
    """
    Preprocess one document.
    :param document: a document is a dict {"text" : "content of document"}
    :param stop_list: stop_list: array of stop words to remove from documents
    :param stop_words: stop_words: boolean True to remove them False otherwise
    :param stemm: stemm: boolean True to stemm words False otherwise
    :param bag_words: bag_words: boolean True to turn documents into bagwords False otherwise
    :return: list of tokens if bagwords False, else dict{"token" : frequency} if bagwords True
    """
    # Tokenize
    REGEX_SPECIAL_CHARACTERS.append(' ')
    REGEX_SPECIAL_CHARACTERS.append('\n')
    tokenizer = Tokenizer(REGEX_SPECIAL_CHARACTERS)
    document_tokenized = tokenizer.tokenize(document)

    # Stop words
    if stop_words:
        new_document = list()
        for word in document_tokenized:
            word = word.lower()
            if word not in stop_list:
                new_document.append(word)
        document_tokenized = new_document

    # Stemmer
    if stemm:
        stemmer = SnowballStemmer("english")
        for i in range(len(document_tokenized)):
            word = document_tokenized[i]
            document_tokenized[i] = stemmer.stem(word)

    # Make bag words dictionary
    if bag_words:
        document_tokenized = make_bag_words(document_tokenized)

    return document_tokenized


def make_bag_words(document_tokenized):
    """
    Turn a list of tokens into a bagwords.
    :param document_tokenized: list of tokens
    :return: dict{"token" : frequency}
    """
    bag_words = dict()
    for token in document_tokenized:
        if token in bag_words.keys():
            bag_words[token] += 1
        else:
            bag_words[token] = 1
    return bag_words


def vocabulary(corpus_tokenized):
    """
    Get a list of each unique word in the whole corpus.
    :param corpus_tokenized: list of bagwords
    :return: a list of each unique token in the whole corpus
    """
    vocab = list()
    for document in corpus_tokenized:
        for word in document:
            if word not in vocab:
                vocab.append(word)
    return vocab
