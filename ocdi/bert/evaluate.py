"""Module to evaluate predictions of a BERT classification model"""
from datetime import timedelta

from ocdi.labels import LABEL_SCHEME_TO_MAPPER
from ocdi.bert.data import DidDataset
from ocdi.bert.predict import predict_with_bert

from ocdi.evaluate import get_scores_from_preds


def evaluate_bert(
        model,
        test_path,
        tokenizer,
        data_prep_opts={},
        col_text="text",
        col_label="dialect",
        test_label_scheme=None,
        train_label_scheme=None,
        batch_size=1,
        num_workers=1,
        only_report=True,
        show_confusion_matrix=False
):
    """
    Evaluate a BERT classification model.

    :param model: BERT classification model
    :param test_path: path to the CSV test dataset
    :param tokenizer: BERT tokenizer
    :param data_prep_opts: options for text preprocessing (see ocdi.svm.preprocessing.preprocess())
    :param col_text: name of the column with text samples in test_path
    :param col_label: name of the column with dialect labels in test_path
    :param test_label_scheme: name of the scheme (see ocdi.labels.SOURCE_TO_SCHEME) to map the testset labels to a common scheme
    :param train_label_scheme: name of the scheme (see ocdi.labels.SOURCE_TO_SCHEME) to map the model labels to a common scheme
    """
    train_labels_mapper = LABEL_SCHEME_TO_MAPPER[train_label_scheme] if train_label_scheme else None
    label2id = model.config.label2id
    # Normalize model labels if needed
    if train_labels_mapper and not all([model_label in list(train_labels_mapper.values()) for model_label in label2id]):
        label2id = {train_labels_mapper[model_label]: label_i for model_label, label_i in label2id.items()}
        model.config.label2id = label2id
        model.config.id2label = {v: k for k, v in label2id.items()}

    ds_test = DidDataset(
        test_path,
        tokenizer,
        col_text=col_text,
        col_label=col_label,
        labels_mapper=LABEL_SCHEME_TO_MAPPER[test_label_scheme],
        label2id=label2id,
        prep_opts=data_prep_opts
    )

    # Get predictions from the model
    preds, time = predict_with_bert(
        model,
        ds_test,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        time=True
    )

    # Get true labels
    true_labels = list(ds_test.data["label_str"])

    results = get_scores_from_preds(
        preds, true_labels,
        labels=sorted(ds_test.label2id),
        only_report=only_report,
        show_confusion_matrix=show_confusion_matrix
    )

    if only_report:
        return preds, true_labels
    else:
        # Compute time stats
        duration = time / timedelta(milliseconds=1)
        results["speed_avg"] = duration / len(preds)

        return results, preds, true_labels
