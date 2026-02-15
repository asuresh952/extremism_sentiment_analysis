# Social Media Extremism Detection - Training Script
# This script trains a single-layer perceptron for extremism detection
# and saves the trained model for export

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from scipy.sparse import hstack, vstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Plot settings
plt.style.use("default")
plt.rcParams["figure.figsize"] = (8, 5)

# Reproducibility
RANDOM_SEED = 30
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH = "extremism_data_final.csv"

# Feature settings
MAX_TOTAL_FEATURES = 20000
N_VADER_FEATURES = 4
MAX_TFIDF_FEATURES = MAX_TOTAL_FEATURES - N_VADER_FEATURES

# Model/training settings
TEST_SIZE = 0.2
BATCH_SIZE = 32
HIDDEN_NEURONS = 512
LR = 0.1
EPOCHS = 127
VAL_THRESHOLD = 0.5

print("="*80)
print("EXTREMISM DETECTION MODEL - TRAINING SCRIPT")
print("="*80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)
df["row_id"] = df.index

print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# 2. ENCODE LABELS
# ============================================================================
print("\n[2/8] Encoding labels...")

label_map = {
    "EXTREMIST": 1,
    "NON_EXTREMIST": 0,
}

df["Binary_Label"] = df["Extremism_Label"].map(label_map).astype(np.int64)
y = df["Binary_Label"].values

print("Label distribution:")
print(df["Extremism_Label"].value_counts())

# ============================================================================
# 3. FIT TF-IDF AND PREPARE VADER
# ============================================================================
print("\n[3/8] Fitting TF-IDF vectorizer...")

texts = df["Original_Message"].fillna("").astype(str).tolist()

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    min_df=3,
    ngram_range=(1, 3)
)
tfidf_vectorizer.fit(texts)

analyzer = VaderAnalyzer()

print(f"Number of TF-IDF features: {len(tfidf_vectorizer.get_feature_names_out())}")

# ============================================================================
# 4. BUILD FEATURE MATRIX
# ============================================================================
print("\n[4/8] Building feature matrix...")

def vectorize_text(text: str):
    """Convert text to feature vector: [TF-IDF | VADER]"""
    X_tfidf = tfidf_vectorizer.transform([text])
    scores = analyzer.polarity_scores(text)
    vader_vec = np.array([[scores["neg"], scores["neu"], scores["pos"], scores["compound"]]])
    X_vader = csr_matrix(vader_vec)
    X_full = hstack([X_tfidf, X_vader], format="csr")
    return X_full

row_vectors = [vectorize_text(t) for t in df["Original_Message"].fillna("").astype(str)]
X = vstack(row_vectors)

print(f"Feature matrix shape: {X.shape}")

# Build feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
vader_feature_names = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
feature_names = tfidf_feature_names + vader_feature_names

print(f"Total features: {len(feature_names)}")

# ============================================================================
# 5. TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n[5/8] Creating train/validation split...")

X_np = X.toarray().astype(np.float32)
y_np = y.astype(np.float32).reshape(-1, 1)

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_np,
    y_np,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_np.reshape(-1),
)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train_np)
y_train = torch.from_numpy(y_train_np)
X_val = torch.from_numpy(X_val_np)
y_val = torch.from_numpy(y_val_np)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Reconstruct validation indices
n_samples = len(df)
all_indices = np.arange(n_samples)
_, val_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

texts_all = df["Original_Message"].fillna("").astype(str).values
texts_val = texts_all[val_idx]

# ============================================================================
# 6. DEFINE MODEL
# ============================================================================
print("\n[6/8] Defining model architecture...")

class SingleLayerNet(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        super(SingleLayerNet, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_size)
        
    def forward(self, x):
        hidden_output = torch.sigmoid(self.hidden_layer(x))
        y_pred = torch.sigmoid(self.output_layer(hidden_output))
        return y_pred

input_size = X_train.shape[1]
output_size = 1

model2 = SingleLayerNet(input_size, HIDDEN_NEURONS, output_size)
print(model2)

def criterion(y_pred, y_true):
    eps = 1e-8
    loss = -1 * (y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))
    return torch.mean(loss)

optimizer = optim.SGD(model2.parameters(), lr=LR)

# ============================================================================
# 7. TRAIN MODEL
# ============================================================================
print("\n[7/8] Training model...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LR}")
print("-"*80)

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_epoch = None
best_state_dict = None

for epoch in range(EPOCHS):
    # TRAIN
    model2.train()
    total_train_loss = 0.0
    total_train_examples = 0

    for xb, yb in train_loader:
        y_pred = model2(xb)
        loss = criterion(y_pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_examples += batch_size

    avg_train_loss = total_train_loss / total_train_examples
    train_losses.append(avg_train_loss)

    # VALIDATION
    model2.eval()
    with torch.no_grad():
        y_val_pred = model2(X_val)
        val_loss = criterion(y_val_pred, y_val).item()
        val_losses.append(val_loss)

        y_val_pred_labels = (y_val_pred >= VAL_THRESHOLD).float()
        correct = (y_val_pred_labels == y_val).sum().item()
        total = y_val.shape[0]
        val_acc = correct / total
        val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_state_dict = model2.state_dict()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

print("-"*80)
print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

# Load best model
model2.load_state_dict(best_state_dict)

# ============================================================================
# 8. EVALUATE AND SAVE MODEL
# ============================================================================
print("\n[8/8] Evaluating and saving model...")

model2.eval()
with torch.no_grad():
    y_val_pred = model2(X_val).cpu().numpy().ravel()

y_val_true = y_val.cpu().numpy().ravel().astype(int)
y_val_pred_labels = (y_val_pred >= VAL_THRESHOLD).astype(int)

acc = accuracy_score(y_val_true, y_val_pred_labels)
f1_macro = f1_score(y_val_true, y_val_pred_labels, average="macro")
f1_weighted = f1_score(y_val_true, y_val_pred_labels, average="weighted")

print("\nFinal Validation Metrics:")
print(f"  Accuracy:      {acc:.4f}")
print(f"  F1 (macro):    {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

# Save everything needed for deployment/analysis
save_dir = "model_output"
os.makedirs(save_dir, exist_ok=True)

# 1. Save PyTorch model
torch.save({
    'epoch': best_epoch,
    'model_state_dict': best_state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
}, os.path.join(save_dir, 'model_checkpoint.pth'))
print(f"\n✓ Model saved to {save_dir}/model_checkpoint.pth")

# 2. Save TF-IDF vectorizer
import pickle
with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"✓ TF-IDF vectorizer saved to {save_dir}/tfidf_vectorizer.pkl")

# 3. Save feature names
with open(os.path.join(save_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feature_names, f)
print(f"✓ Feature names saved to {save_dir}/feature_names.pkl")

# 4. Save configuration
config = {
    'RANDOM_SEED': RANDOM_SEED,
    'MAX_TOTAL_FEATURES': MAX_TOTAL_FEATURES,
    'N_VADER_FEATURES': N_VADER_FEATURES,
    'MAX_TFIDF_FEATURES': MAX_TFIDF_FEATURES,
    'TEST_SIZE': TEST_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    'HIDDEN_NEURONS': HIDDEN_NEURONS,
    'LR': LR,
    'EPOCHS': EPOCHS,
    'VAL_THRESHOLD': VAL_THRESHOLD,
    'input_size': input_size,
    'output_size': output_size,
}
with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)
print(f"✓ Configuration saved to {save_dir}/config.pkl")

# 5. Save training history as CSV
history_df = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})
history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
print(f"✓ Training history saved to {save_dir}/training_history.csv")

# 6. Save validation data for SHAP analysis
np.save(os.path.join(save_dir, 'X_train_np.npy'), X_train_np)
np.save(os.path.join(save_dir, 'X_val_np.npy'), X_val_np)
np.save(os.path.join(save_dir, 'y_val_true.npy'), y_val_true)
np.save(os.path.join(save_dir, 'y_val_pred.npy'), y_val_pred)
np.save(os.path.join(save_dir, 'val_idx.npy'), val_idx)
print(f"✓ Validation data saved for SHAP analysis")

# 7. Save metrics summary
metrics_summary = {
    'best_epoch': best_epoch,
    'best_val_acc': best_val_acc,
    'final_accuracy': acc,
    'final_f1_macro': f1_macro,
    'final_f1_weighted': f1_weighted,
}
with open(os.path.join(save_dir, 'metrics_summary.pkl'), 'wb') as f:
    pickle.dump(metrics_summary, f)
print(f"✓ Metrics summary saved to {save_dir}/metrics_summary.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nAll files saved to: {save_dir}/")
print("\nSaved files:")
print("  - model_checkpoint.pth       (PyTorch model)")
print("  - tfidf_vectorizer.pkl       (TF-IDF vectorizer)")
print("  - feature_names.pkl          (Feature names list)")
print("  - config.pkl                 (All configuration)")
print("  - training_history.csv       (Loss/accuracy curves)")
print("  - X_train_np.npy            (Training features)")
print("  - X_val_np.npy              (Validation features)")
print("  - y_val_true.npy            (True labels)")
print("  - y_val_pred.npy            (Predicted probabilities)")
print("  - val_idx.npy               (Validation indices)")
print("  - metrics_summary.pkl        (Performance metrics)")

print("\n" + "="*80)
print("USAGE - Load model in new script:")
print("="*80)
print("""
import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
checkpoint = torch.load('model_output/model_checkpoint.pth')
model = SingleLayerNet(input_size, hidden_neurons, output_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load vectorizer
with open('model_output/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load config
with open('model_output/config.pkl', 'rb') as f:
    config = pickle.load(f)

# Make predictions on new text
def predict_text(text):
    X_tfidf = tfidf_vectorizer.transform([text])
    scores = analyzer.polarity_scores(text)
    vader_vec = np.array([[scores["neg"], scores["neu"], scores["pos"], scores["compound"]]])
    X_vader = csr_matrix(vader_vec)
    X_full = hstack([X_tfidf, X_vader], format="csr")
    X_tensor = torch.from_numpy(X_full.toarray().astype(np.float32))
    with torch.no_grad():
        prob = model(X_tensor).item()
    return prob
""")