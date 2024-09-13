"""Functions for loading and preprocessing data before training models or running inference"""

from copy import deepcopy
from functools import partial
from string import punctuation

import regex as re

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, strip_accents_unicode, strip_accents_ascii

import occitok

WORDS_PIPELINE_STR = "words_pipeline"
CHARS_PIPELINE_STR = "chars_pipeline"
WORD_TRANSFORMER_KEY = "word_ngrams"
CHAR_TRANSFORMER_KEY = "char_ngrams"

DEFAULT_STOPWORDS = [
    "occitan", "franc", "franç",
    "aranés", "aranes", "aranais",
    "lenga", "langu", "lengadocian", "languedocien",
    "gascon", "lemosin", "limousin",
    "provença", "provenca",
    "auvern",
    "vivarés", "vivares", "vivaro", "alpin", "aupenc", "vivaro-alpin",
    "niçois", "niçar", "nicar", "nicois", "nissar"
]


def tokenize_text(text):
    """Performs extra preprocessing steps + splits a string into tokens using occitok

    :param text: str. input text to tokenize
    :param remove_punctuation: True/False or "keep_apos" to remove all punctuation except apostrophes
    :param remove_numbers: True/False to remove words that contain only digits

    :return: tokenized text as a list of words
    """

    # Tokenize text into words with Occitok
    words = occitok.tokenize_line(text)

    return words


def remove_short_samples(df, text_col_name="text", threshold=3):  # unused in our experiments.
    return df[df[text_col_name].apply(tokenize_text).apply(len) > threshold]


def select_vectorizer_type(vectorizer_params):
    """
    Get the vectorizer class corresponding to the given string in the dict of params.
    """
    vec_type = vectorizer_params["vectorizer_type"]
    if vec_type == "count":
        vec_class = CountVectorizer
    elif vec_type == "tfidf":
        vec_class = TfidfVectorizer
    else:
        raise ValueError(f"vectorizer_type '{vec_type}' not implemented")

    del vectorizer_params["vectorizer_type"]
    return vec_class


def preprocess_list(tokens, lowercase=False, strip_accents=False):
    """Preprocess a list of tokens with lowercase and/or stripping accents"""
    tokens = [
        preprocess(
            tok,
            lowercase=lowercase,
            strip_accents=strip_accents
        )
        for tok in tokens
    ]
    return tokens


def preprocess_for_chars(s, lowercase=False, strip_accents=False):
    s = preprocess(
        s,
        lowercase=lowercase,
        strip_accents=strip_accents
    )
    return s


def preprocess(
        s,
        lowercase=False,
        strip_accents=False,
        remove_punctuation=False,
        norm_dup_chars=False,
        mask_urls=False,
        mask_emails=False,
        mask_numbers=False,
        stopwords_list=None
):
    """
    Applies preprocessing steps to a string.

    :param s: str. text to preprocess
    :param lowercase: True/False
    :param strip_accents: False or "ascii" or "unicode" (calling scikit-learn functions)
    :param remove_punctuation: True/False or "keep_apos" to remove all punctuation except apostrophes
        or "punct" to remove only punctuation in string.punctuation
    :param norm_dup_chars: False/True to replace the found sequences of duplicated characters (3+) to only 1 character
    :param mask_urls: False or "remove" or str to replace detected URLs with
    :param mask_emails: False or "remove" or str to replace detected emails with
    :param mask_numbers: False or "remove" or str to replace detected numbers with
    :param stopwords_list: list of words to remove before tokenization ,
        i.e. they will not be used in word nor char vocabs.
        Note that the matching is done from start of word,
        with no right-side boundary (so "lenga" matches also "lengadoc").
        Pass "default" to use the default stopwords list defined above (DEFAULT_STOPWORDS)
    :return: preprocessed string
    """
    # Handle URLs
    if mask_urls is not False:
        assert isinstance(mask_urls, str), f"mask_urls value must be False or a string to replace detected urls with"
        repl = "" if mask_urls == "remove" else mask_urls
        s = re.sub(
            # Regex pattern
            "(([A-Za-z]{3,9}:(?:\/\/)|www\.)"  # protocol
            "[A-Za-z0-9.-]+((?:\/[-\+~%\/.\w_]*)?\??"  # main address
            "(?:[-\+=&;%@.\w_]*)#?(?:[.\!\/\\w]*))?)",  # extra parameters (starting with '?')
            # Replacement = remove
            repl,
            # Input text
            s
        )

    # Handle email addresses
    if mask_emails is not False:
        assert isinstance(mask_emails, str), \
            f"mask_emails value must be False or a string to replace detected emails with"
        repl = "" if mask_emails == "remove" else mask_emails
        s = re.sub(
            # Regex pattern
            r"[\w\d][^\s]+@[^\s]+[^\s\W]", # very basic and prone to false positives, yet words containing "@" are probably noise anyway for any LID task
            # Replacement = remove
            repl,
            # Input text
            s
        )

    # Handle accents
    if strip_accents:
        if strip_accents == "unicode":
            f_accents = strip_accents_unicode
        elif strip_accents == "ascii":
            f_accents = strip_accents_ascii
        else:
            raise ValueError(f"Unrecognized mode for stripping accents: {strip_accents}")
        s = f_accents(s)

    # Handle duplicated chars
    if norm_dup_chars:
        s = re.sub(r"(\w)\1{2,}", r"\1", s)

    # Remove punctuation
    # Configure punctuation set of characters
    if remove_punctuation == "keep_apos":
        # Remove apostrophe from the punctuation list
        punct_symbols = punctuation.replace("'", "")
    elif remove_punctuation == "punct":
        punct_symbols = punctuation
    else:  # False
        punct_symbols = None

    if remove_punctuation:
        if punct_symbols:
            s = re.sub(r"\L<symbols>", " ", s, symbols=punct_symbols)
        else:
            s = re.sub(r"\W|_", " ", s)

    # Remove words containing only digits
    if mask_numbers is not False:
        assert isinstance(mask_numbers, str), \
            f"mask_numbers value must be False or a string to replace detected numbers with"
        repl = " " if mask_numbers == "remove" else mask_numbers
        s = re.sub(r"\d+", repl, s)

    # Lowercasing
    if lowercase:
        s = s.lower()

    if stopwords_list:
        if stopwords_list == "default":
            stopwords_list = DEFAULT_STOPWORDS
        assert isinstance(stopwords_list, list), f"stopwords_list should be a list (found {type(stopwords_list)})"
        s = re.sub(r"\b\L<stopwords>\w*", " ", s, stopwords=stopwords_list, flags=re.IGNORECASE)

    return s


def make_words_pipeline(tokenizer, vectorizer_params):
    """
    Creates the Transformer that vectorizes the texts into word n-grams
    :param tokenizer: function. to tokenize each sentence into words
    :param vectorizer_params: dict. kwargs for the CountVectorizer object
    :return: Transformer object
    """
    # Get the vectorizer class (CountVectorizer/TfidfVectorizer)
    vec_class = select_vectorizer_type(vectorizer_params)

    if vectorizer_params.get("input_is_list"):
        if "lowercase" in vectorizer_params or "strip_accents" in vectorizer_params:
            vectorizer_params["preprocessor"] = partial(
                preprocess_list,
                lowercase=vectorizer_params.get("lowercase"),
                strip_accents=vectorizer_params.get("strip_accents")
            )
        del vectorizer_params["input_is_list"]

    # Creation of the vectorizer object
    word_vectorizer = vec_class(
        analyzer='word',
        tokenizer=tokenizer,
        **vectorizer_params
    )

    return word_vectorizer


def make_chars_pipeline(vectorizer_params):
    """
    Creates the Transformer that vectorizes the texts into character ngrams (inside of word boundaries)
    :param vectorizer_params: dict. kwargs for the CountVectorizer object
    :return: Transformer object
    """
    # Get the vectorizer class (CountVectorizer/TfidfVectorizer)
    vec_class = select_vectorizer_type(vectorizer_params)

    # Define preprocessing function
    if "preprocessor" not in vectorizer_params:
        if "lowercase" in vectorizer_params or "strip_accents" in vectorizer_params:
            vectorizer_params["preprocessor"] = partial(
                preprocess_for_chars,
                lowercase=vectorizer_params.get("lowercase"),
                strip_accents=vectorizer_params.get("strip_accents")
            )

    # Creation of the vectorizer object
    char_vectorizer = vec_class(analyzer='char_wb', **vectorizer_params)

    return char_vectorizer


def make_preprocess_pipeline(
        preprocessing_params,
        tokenizer,
        col_words_pipeline="text",
        col_chars_pipeline="text"
):
    """
    Creates a preprocess pipeline with specific hyperparameters.
    :param preprocessing_params: dict. params to build the words and/or chars pipelines,
        in the format: {WORDS_PIPELINE_STR: {"vectorizer_type": ("tfidf" or "count"), **vectorizer_params (except analyzer and tokenizer)}}
    :param tokenizer: tokenizer function to use for extracting word features (or None to use the default one in sklearn)
    :param col_words_pipeline: DataFrame column to use for extracting word features (default "text")
    :param col_chars_pipeline: DataFrame column to use for extracting character features (default "text")
    :return: list : 0 = the entire preprocessing pipeline.
                    1 = word n-grams vectorizer
                    2 = character n-grams vectorizer
    """
    transformers_list = []
    preprocessing_params = deepcopy(preprocessing_params)

    if WORDS_PIPELINE_STR in preprocessing_params:
        words_vectorizer = make_words_pipeline(
            tokenizer=tokenizer,
            vectorizer_params=preprocessing_params[WORDS_PIPELINE_STR]
        )
        words_transformer = ColumnTransformer(
            [("ct", words_vectorizer, col_words_pipeline)],
            verbose_feature_names_out=False
        )
        transformers_list.append((WORD_TRANSFORMER_KEY, words_transformer))

    if CHARS_PIPELINE_STR in preprocessing_params:
        chars_vectorizer = make_chars_pipeline(preprocessing_params[CHARS_PIPELINE_STR])
        chars_transformer = ColumnTransformer(
            [("ct", chars_vectorizer, col_chars_pipeline)],
            verbose_feature_names_out=False
        )
        transformers_list.append((CHAR_TRANSFORMER_KEY, chars_transformer))

    union = FeatureUnion(transformer_list=transformers_list)

    preprocess_pipeline = make_pipeline(union)
    words_transformer = union.named_transformers.get(WORD_TRANSFORMER_KEY)
    chars_transformer = union.named_transformers.get(CHAR_TRANSFORMER_KEY)
    return preprocess_pipeline, words_transformer, chars_transformer
