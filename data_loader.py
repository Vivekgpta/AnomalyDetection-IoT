# Functions for loading and preprocessing the dataset

from datasets import load_dataset
dataset = load_dataset("Mireu-Lab/UNSW-NB15")
import pandas as pd
df = pd.DataFrame(dataset['train'])
df.to_csv("UNSW-NB15.csv", index=False)
