"""Functions for analyzing/evaluating fitted models"""
from datetime import datetime, timedelta

from IPython.display import display
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE, SelectFromModel

from ocdi.evaluate import get_scores_from_preds


def get_top_features(pipeline, n=10):

    # Get feature names from the transformers
    feature_transformers = pipeline.named_steps["preprocess"].named_steps["featureunion"].named_transformers
    vocab = []
    for transformer_name, transformer in feature_transformers.items():
        vocab_part = transformer.get_feature_names_out()
        if transformer_name.startswith("c"):
            vocab_part = ["c_" + v.replace(" ", "∅") for v in vocab_part]
        else:
            vocab_part = ["w_" + v for v in vocab_part]
        if len(vocab_part) > 0:
            vocab.extend(vocab_part)

    # Get feature weights from the estimator
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, CalibratedClassifierCV):
        if len(clf.calibrated_classifiers_) > 1:
            raise NotImplementedError
        clf = clf.calibrated_classifiers_[0].estimator
    if isinstance(clf, RFE) or isinstance(clf, SelectFromModel):
        clf = clf.estimator
    weights_matrix = clf.coef_

    # For each dialect, extract the features with the largest weights
    top_features_per_dialect = {}
    dialects = sorted(clf.classes_)
    for dia in dialects:
        # Get row of the weights that corresponds to the dialect
        class_index = list(clf.classes_).index(dia)
        if len(dialects) == 2 and class_index == 1:  # Binary = single list of weights
            class_index = 0
        # Match feature names and weights
        dia_weights = {ft_name: w for ft_name, w in zip(vocab, weights_matrix[class_index])}
        assert len(dia_weights) == len(vocab), f"{len(dia_weights)} != {len(vocab)}"
        # Sort features by weight
        dia_weights = dict(sorted(dia_weights.items(), key=lambda item: item[1], reverse=True))
        # Select n top features
        top_features_dia = [ft_name for ft_name in dia_weights.keys()][:n]
        top_features_per_dialect[dia] = top_features_dia

    return top_features_per_dialect


def infos_features(pipeline, show_n_features):
    def infos_features_one_transformer(transformer, clf, show_n, feature_type):
        """
        Prints size of vocabulary (features) and displays most important features for each dialect
        :param transformer: Transformer object for transforming sentences into word or character n-grams
        :param clf: classifier (with .coef_)
        :param show_n: int. number of vocab entries to print as examples
        :param feature_type: str. name to show for the type of feature ("word"/"chars")
        """
        vocab = transformer.get_feature_names_out()
        if feature_type.startswith("c"):
            # char_wb vectorizer mode pads word boundaries with whitespace in order to distinguish affixes
            # yet the whitespaces won't be visible in the displays of pandas DataFrames,
            # therefore we replace them with a more visible character
            vocab = [v.replace(" ", "#") for v in vocab]

        print(f"Vocabulary size for {feature_type}: {len(vocab)}")

        # Show most important features per dialect
        if show_n:
            # Select weights corresponding for word or character features
            if feature_type.lower().startswith("w"):
                weights_matrix = clf.coef_[:, :len(vocab)]
            else:
                weights_matrix = clf.coef_[:, -len(vocab):]
            # For each dialect, build a dataframe associating each feature with its weight
            ft_weights_per_dialect = []
            dialect_names = sorted(clf.classes_)
            for dia_name in dialect_names:
                class_i = list(clf.classes_).index(dia_name)
                if len(dialect_names) == 2 and class_i == 1:  # Binary = single list of weights
                    class_i = 0
                dia_weights = {ft_name: w for ft_name, w in zip(vocab, weights_matrix[class_i])}
                assert len(dia_weights) == len(vocab)
                df_dia_weights = pd.DataFrame.from_dict(dia_weights, orient="index", columns=["weight_" + dia_name])
                ft_weights_per_dialect.append(df_dia_weights)
            # Merge all dialect-specific dataframes
            df_weights = pd.concat(ft_weights_per_dialect, axis=1)
            # Compute total weight of each feature as the sum of absolute values for each dialect
            df_weights["weight_total_absolute"] = df_weights.apply(
                lambda row: sum([abs(row[col]) for col in df_weights.columns]),
                axis=1
            )
            df_weights.sort_values("weight_total_absolute", ascending=False, inplace=True)

            # Display top 10 features for each dialect
            for dia_name in dialect_names:
                df_weights_sorted_dia = df_weights.sort_values("weight_" + dia_name, ascending=False)
                print(f"Top {show_n} {feature_type} features for {dia_name}")
                display(df_weights_sorted_dia[:show_n])

    clf = pipeline.named_steps["clf"]
    if isinstance(clf, CalibratedClassifierCV):
        if len(clf.calibrated_classifiers_) > 1:
            raise NotImplementedError
        clf = clf.calibrated_classifiers_[0].estimator
    if isinstance(clf, RFE) or isinstance(clf, SelectFromModel):
        clf = clf.estimator

    feature_transformers = pipeline.named_steps["preprocess"].named_steps["featureunion"].named_transformers

    for transformer_name, transformer in feature_transformers.items():
        infos_features_one_transformer(
            transformer,
            clf,
            show_n_features,
            transformer_name
        )


def evaluate_model(trained_pipeline,
                   test_df,
                   return_preds=False,
                   show_confusion_matrix=True,
                   label_col="label"):
    def get_clf_classes(trained_pipeline):
        clf = trained_pipeline.named_steps["clf"]
        if isinstance(clf, CalibratedClassifierCV):
            if len(clf.calibrated_classifiers_) > 1:
                raise NotImplementedError
            clf = clf.calibrated_classifiers_[0].estimator
        if isinstance(clf, SelectFromModel):
            clf = clf.estimator_
        return clf.classes_

    # Predict labels for all texts of the test data
    start_time = datetime.now()
    y_preds = trained_pipeline.predict(test_df)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Computed all predictions in {duration}")

    # Get true labels to compare with
    y_true = test_df[label_col]

    results = get_scores_from_preds(
        y_preds,
        y_true,
        only_report=False,
        show_confusion_matrix=show_confusion_matrix
    )

    # Compute time stats
    duration = duration / timedelta(milliseconds=1)
    results["speed_avg"] = duration / len(test_df)

    if not return_preds:
        return results
    else:
        return results, y_preds
