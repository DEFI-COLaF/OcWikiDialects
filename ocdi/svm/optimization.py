"""Functions to optimize hyperparameters of a Pipeline"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from ocdi.svm.preprocessing import make_preprocess_pipeline, tokenize_text


def optimize_params(params_to_optimize, preprocessing_params, model, train_df, n_jobs):
    # Build pipeline
    preprocess_pipeline, words_pipeline, chars_pipeline = make_preprocess_pipeline(
        preprocessing_params, tokenizer=tokenize_text
    )
    classifier_pipeline = Pipeline([("preprocess", preprocess_pipeline),
                                    ("clf", model)])

    # Setup optimization
    optimizing_pipeline = GridSearchCV(
        classifier_pipeline,
        params_to_optimize,
        scoring="f1_macro",
        return_train_score=True,
        n_jobs=n_jobs,
        refit=False,
        cv=5,
        error_score="raise"
    )

    # Launch optimization runs
    optimizing_pipeline.fit(train_df, train_df["label"])

    # Report on results
    print(f"Best params: {optimizing_pipeline.best_params_}")

    return optimizing_pipeline
