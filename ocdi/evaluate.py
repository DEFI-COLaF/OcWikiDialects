"""Module with function to evaluate based on any list of preds and true labels"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sklearn_metrics


def group_other_preds(preds, test_labels, other_label="other"):
    grouped_preds = []
    for pred in preds:
        if pred in test_labels:
            grouped_preds.append(pred)
        else:
            grouped_preds.append(other_label)
    return grouped_preds


def get_scores_from_preds(preds, refs, labels=None,
                          only_report=True,
                          show_confusion_matrix=False):
    # Group "other" labels to avoid overpenalizing models when they predict out of the test set classes
    test_labels = set(refs)
    preds = group_other_preds(preds, test_labels)

    if not labels:
        labels = sorted(test_labels.union(set(preds)))

    # Print classification report
    report = sklearn_metrics.classification_report(
        refs, preds,
        zero_division=np.nan,
        labels=labels,
        digits=4
    )
    print(f"\n\nClassification report:\n{report}\n\n")

    if show_confusion_matrix:
        # Compute confusion matrix
        cm = sklearn_metrics.confusion_matrix(
            y_true=refs,
            y_pred=preds,
            normalize="true",
            labels=labels
        ) * 100
        # Show confusion matrix
        cm_display = sklearn_metrics.ConfusionMatrixDisplay(
            cm,
            display_labels=[l.upper() for l in labels]
        )
        cm_display.plot(values_format=".2f", im_kw={"vmin": 0, "vmax": 100})
        plt.show()

    if only_report:
        return None

    # Compute accuracy, precision, recall, f1 with macro average
    accuracy = sklearn_metrics.accuracy_score(refs, preds)
    # accuracy_balanced = sklearn_metrics.balanced_accuracy_score(refs, preds)  # same as recall-macro (per definition, cf. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
    precision, recall, f1_score, _ = sklearn_metrics.precision_recall_fscore_support(
        refs, preds,
        zero_division=np.nan,
        average="macro"
    )

    # Get f1 per class
    f1_per_class = sklearn_metrics.f1_score(
        refs, preds,
        zero_division=np.nan,
        average=None,
        labels=labels
    )
    f1_per_class = {"f1_" + label: score for label, score in zip(labels, f1_per_class)}

    # Get precision per class
    p_per_class = sklearn_metrics.precision_score(
        refs, preds,
        zero_division=np.nan,
        average=None,
        labels=labels
    )
    p_per_class = {"precision_" + label: score for label, score in zip(labels, p_per_class)}

    # Get recall per class
    r_per_class = sklearn_metrics.recall_score(
        refs, preds,
        zero_division=np.nan,
        average=None,
        labels=labels
    )
    r_per_class = {"recall_" + label: score for label, score in zip(labels, r_per_class)}

    # Store results
    results = {
        "f1_macro": f1_score,
        "accuracy": accuracy,
        # "accuracy_balanced": accuracy_balanced, # same as recall-macro
        "precision_macro": precision,
        "recall_macro": recall,
        **f1_per_class,
        **p_per_class,
        **r_per_class
    }

    return results


def get_baseline_theoretic_scores_uniform(df_test):
    dialect_distribution = df_test.value_counts("dialect").sort_index()
    dialects = list(dialect_distribution.index)
    n_dialects = len(dialects)
    test_size = len(df_test)

    # Compute accuracy and recall -- unaffected by class frequencies
    accuracy = 1 / n_dialects
    recall = 1 / n_dialects

    # Compute recall and precision and F1 per dialect + F1 macro averaged
    precision_per_class = dialect_distribution / test_size  # Depends on the class frequency
    f1_per_class = 2 * (precision_per_class * recall) / (precision_per_class + recall)

    # Compute averaged values
    precision = precision_per_class.mean()
    f1_score = f1_per_class.mean()

    # Store results
    r_per_class = {"recall_" + label: recall for label in dialects}
    p_per_class = {"precision_" + label: score for label, score in zip(dialects, precision_per_class)}
    f1_per_class = {"f1_" + label: score for label, score in zip(dialects, f1_per_class)}

    results = {
        "f1_macro": f1_score,
        "accuracy": accuracy,
        # "accuracy_balanced": accuracy_balanced,  # same as recall-macro (per definition, cf. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
        "precision_macro": precision,
        "recall_macro": recall,
        **f1_per_class,
        **p_per_class,
        **r_per_class
    }

    return results

