"""Utils for predicting dialect labels with a trained BERT Did model"""
from datetime import datetime

from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ocdi.bert.data import DidDataset


def predict_with_bert(model, did_dataset: DidDataset, tokenizer, batch_size=1, num_workers=1, time=False):
    """Predicts dialect labels for each sample of the torch dataset.
    The data"""
    # Set a DataLoader to predict by batches
    dataloader = DataLoader(
        did_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=num_workers
    )

    device = next(model.parameters()).device
    print(f"Model for predictions is on device: {device}")

    # Get predictions
    start = datetime.now()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {}
            for col in ["input_ids", "attention_mask"]:
                if col in batch:
                    inputs[col] = batch[col].to(device)
            logits = model(**inputs).logits
            batch_preds_indices = logits.argmax(dim=1).tolist()
            batch_preds = [model.config.id2label[i] for i in batch_preds_indices]
            preds.extend(batch_preds)
    end = datetime.now()

    if time:
        duration = end - start
        return preds, duration
    else:
        return preds
