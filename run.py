# Main runner script to load data, train and evaluate model

# from src.data_loader import load_unsw_dataset
from src.data_loader import load_dataset
# from datasets import load_dataset


from src.model import build_dnn
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load dataset
print("Loading dataset...")
# df = load_unsw_dataset()
# df = load_dataset()
# df = load_dataset("Mireu-Lab/UNSW-NB15")
dataset = load_dataset("Mireu-Lab/UNSW-NB15")
df = dataset['train'].to_pandas()  # Optional: convert to Pandas DataFrame

# dataset = load_dataset("Mireu-Lab/UNSW-NB15")
# print(dataset)
# print(dataset['train'][0])  # Show first training example

# Step 2: Preprocessing (simplified)
X = df.drop(columns=["label"])  # Replace with actual feature columns
y = df["label"]

# Ensure numeric input
X = pd.get_dummies(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build model
model = build_dnn(input_dim=X_train.shape[1])

# Step 5: Train
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Step 6: Predict and evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
evaluate_model(y_test, y_pred)
