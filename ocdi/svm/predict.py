"""Module with function to predict labels for a list of samples"""

import pandas as pd

def predict_with_svm(model, samples):
    df = pd.DataFrame({"text": samples})
    preds = model.predict(df)
    return preds
