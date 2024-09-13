"""Module to evaluate predictions of a FastText classification model"""
from datetime import datetime, timedelta

import fasttext
from tqdm.auto import tqdm

from ocdi.evaluate import get_scores_from_preds


def trim_fasttext_label(label):
    return label.replace("__label__", "")


def predict_label(text, fasttext_model):
    start = datetime.now()
    label, score = fasttext_model.predict(text)
    end = datetime.now()
    duration = (end - start) / timedelta(milliseconds=1)
    return label[0], score[0], duration


def evaluate_fasttext(model_path, df_test, only_report=False, show_confusion_matrix=True):
    """
    Run predictions and evaluate based on true labels in df_test

    :param model_path: path to the fasttext model (.bin)
    :param df_test: pandas DataFrame with columns "text" and "label"

    :return: dict with scores (accuracy, macro-averaged f1/precision/recall, f1/precision/recall per class)
    """
    # Load model
    fasttext_model = fasttext.load_model(model_path)

    # Get predictions
    tqdm.pandas()
    df_results = df_test.progress_apply(
        lambda row: predict_label(row["text"], fasttext_model),
        result_type="expand",
        axis=1
    )
    df_results.rename(columns={0: "label", 1: "score", 2: "runtime"}, inplace=True)

    # Get true and pred labels
    y_preds = df_results["label"].apply(trim_fasttext_label)
    true_labels = df_test["label"].apply(trim_fasttext_label)
    labels_unique = [trim_fasttext_label(label) for label in sorted(df_test["label"].unique())]

    # Compute scores
    results = get_scores_from_preds(
        preds=y_preds,
        refs=true_labels,
        labels=labels_unique,
        only_report=only_report,
        show_confusion_matrix=show_confusion_matrix
    )

    # Compute stats on the score column
    results["score_avg"] = df_results["score"].mean()
    results["score_std"] = df_results["score"].std()

    # Compute stats on time
    results["runtime_avg"] = df_results["runtime"].mean()
    results["runtime_std"] = df_results["runtime"].std()

    return results
