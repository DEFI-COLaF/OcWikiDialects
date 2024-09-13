"""Module to handle saving and loading LIDoc models"""

import pickle

from sklearn.pipeline import Pipeline


def save_sklearn_pipeline(pipeline, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sklearn_pipeline(path):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    assert isinstance(pipeline, Pipeline), "pickle file doesn't contain a Pipeline object"
    return pipeline
