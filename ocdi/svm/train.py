"""Functions for fitting models and running full pipelines from preprocessing to evaluation"""

import pprint
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ocdi.constants import WORDS_PIPELINE_STR, CHARS_PIPELINE_STR, SHOW_N_FEATURES_DEFAULT
from ocdi.svm.preprocessing import make_preprocess_pipeline, tokenize_text
import ocdi.svm.evaluation as evaluation


def train_model(preprocess_pipeline, model, train_df, label_col="label"):
    """
    Fits a preprocessing+classification pipeline on the given training data

    :param preprocess_pipeline: Pipeline object for the preprocessing transformations
    :param model: classifier object (implementing fit()) to train
    :param train_df: DataFrame with columns "text" and "label", to fit data with
    :return: trained Pipeline object
    """
    classifier_pipeline = Pipeline([("preprocess", preprocess_pipeline),
                                    ("clf", model)])

    start_time = datetime.now()
    classifier_pipeline.fit(train_df, train_df[label_col])
    end_time = datetime.now()
    print(f"Pipeline fitting done in {str(end_time - start_time)}")

    return classifier_pipeline


def build_train_evaluate_model(
        preprocess_params, model, train_df, test_df,
        tokenizer=tokenize_text,
        show_n_features=SHOW_N_FEATURES_DEFAULT,
        show_confusion_matrix=True
):
    """
    Build and fit the entire pipeline, from preprocessing to evaluation
    :param preprocess_params: dict with params for the words and/or chars vectorizers
    :param model: instance of the model to train
    :param train_df: DataFrame with the training samples
    :param test_df: DataFrame with the samples to evaluate on
    :param tokenizer: function to tokenize text into tokens
    :param show_n_features: int. number of features (for each word/char and dialect) to display
    :param show_confusion_matrix: bool. False to not compute/display the confusion matrix during evaluation
    :return: trained model
    """
    # Build preprocessing pipeline
    preprocess_pipeline, words_transformer, chars_transformer = make_preprocess_pipeline(
        preprocess_params, tokenizer=tokenizer
    )

    # Train model
    trained_pipeline = train_model(preprocess_pipeline, model, train_df)

    # Show info on extracted features used during training
    evaluation.infos_features(trained_pipeline, show_n_features)

    # Evaluate model
    results = evaluation.evaluate_model(trained_pipeline, test_df, show_confusion_matrix=show_confusion_matrix)

    return trained_pipeline, results


def train_n_times(
        preprocess_params, model_class, train_df, test_df,
        n_times, best_on="accuracy_balanced",
        seeds=None, model_kwargs=None,
        show_n_features=SHOW_N_FEATURES_DEFAULT,
        show_confusion_matrix=True
):
    """
    Train a classifier n times, evaluates each of them, compute the average scores and returns the best model
    :param preprocess_params: dict with params for the words and/or chars vectorizers
    :param model_class: class of the model to train (a new instance will be initialized for each run),
        or function that returns a model instance to be trained (implements fit() and predict())
    :param train_df: DataFrame with the training samples
    :param test_df: DataFrame with the samples to evaluate on
    :param n_times: int. number of models to train
    :param best_on: str. metric based on which to choose the best model (default "accuracy_balanced")
    :param seeds: list of int seeds to use for each run (default None = random)
    :param model_kwargs: kwargs to pass to the model_class init method
    :param show_n_features: int. number of features (for each word/char and dialect) to display
    :param show_confusion_matrix: bool. False to not compute/display the confusion matrix during evaluation
    :return: best model, average scores dict
    """
    all_models = []
    all_results = []

    if seeds:
        assert len(seeds) == n_times, (f"Number of seeds must match the number of runs to launch "
                                       f"({len(seeds)} != {n_times}")

    # Train models
    for run_i in range(n_times):
        print(f"== Run {run_i+1}/{n_times} ==")
        # Initialize model
        if not model_kwargs:
            model_kwargs = {}
        if seeds:
            model_kwargs["random_state"] = np.random.RandomState(seeds[run_i])
        model = model_class(**model_kwargs)
        # Fit model
        model, res = build_train_evaluate_model(
            deepcopy(preprocess_params),
            model,
            train_df,
            test_df,
            show_n_features=show_n_features if run_i == 0 else 0,
            show_confusion_matrix=show_confusion_matrix if run_i == 0 else False
        )
        all_models.append(model)
        all_results.append(res)

    df_results = pd.DataFrame(all_results)

    # Choose the best model
    best_model = all_models[np.argmax(df_results[best_on])]

    # Compute scores averages
    results_average = df_results.mean().to_dict()
    print(f"Average results:")
    pprint.pp(results_average)

    return best_model, results_average


def build_train_evaluate_model_pos(
        preprocess_params, model, train_df, test_df_gold_pos, test_df_pred_pos,
        col_raw_text="text", col_pos_text="text_pos", show_n_features=SHOW_N_FEATURES_DEFAULT):
    """
    Build and fit the entire pipeline, from preprocessing to evaluation
    :param preprocess_params: dict with params for the words and/or chars vectorizers
    :param model: instance of the model to train
    :param train_df: DataFrame with the training samples
    :param test_df_gold_pos: DataFrame with the samples to evaluate on, where POS tags are GOLD
    :param test_df_pred_pos: DataFrame with the samples to evaluate on, where POS tags are PREDICTED
    :param col_raw_text: DataFrame column name with the raw samples text
    :param col_pos_text: DataFrame column name where the samples are lists of merged POS+token strings
    :param show_n_features: int. number of features (for each word/char and dialect) to display
    :return: trained model
    """
    # Build preprocessing pipeline
    preprocess_pipeline, words_transformer, chars_transformer = make_preprocess_pipeline(
        preprocess_params,
        tokenizer=lambda tok_list: tok_list,
        col_words_pipeline=col_pos_text,
        col_chars_pipeline=col_raw_text
    )

    # Train model
    clf = train_model(preprocess_pipeline, model, train_df)

    # Show info on extracted features used during training
    if WORDS_PIPELINE_STR in preprocess_params:
        evaluation.infos_features(words_transformer, clf.named_steps["clf"], show_n_features, "words")
    if CHARS_PIPELINE_STR in preprocess_params:
        evaluation.infos_features(chars_transformer, clf.named_steps["clf"], show_n_features, "chars")

    # Evaluate model on test_df with gold POS
    print("Results on test_df with GOLD POS")
    results_goldpos = evaluation.evaluate_model(clf, test_df_gold_pos)

    # Evaluate model on test_df with predicted POS
    print("Results on test_df with PREDICTED POS")
    results_predpos = evaluation.evaluate_model(clf, test_df_pred_pos)

    return clf, results_goldpos, results_predpos
