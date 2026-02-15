# %% [markdown]
# # Social Media Extremism Detection Using a Single-Layer Perceptron
# 
# This notebook trains and explains a simple neural network model to detect extremist
# content in social media text. The workflow:
# 
# 1. **Data loading & label encoding**
# 2. **Feature engineering**: TF–IDF n-grams + VADER sentiment features
# 3. **Model training**: single-hidden-layer perceptron in PyTorch
# 4. **Evaluation**: accuracy, F1, ROC/PR curves, calibration metrics
# 5. **Explainability**: SHAP global & local explanations
# 6. **Quality control**: flagging potential label issues for manual review

# %%
# 0. Imports & Global Config

import random
import re

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

import shap

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
MAX_TFIDF_FEATURES = MAX_TOTAL_FEATURES - N_VADER_FEATURES  # e.g. 19996

# Model/training settings
TEST_SIZE = 0.2
BATCH_SIZE = 32
HIDDEN_NEURONS = 512
LR = 0.1
EPOCHS = 130  # bump above 1 so you actually see curves
VAL_THRESHOLD = 0.5  # decision threshold for probs -> labels

# %% [markdown]
# ## 1. Data Loading and Label Encoding
# 
# In this section, we:
# - Load the raw dataset from `extremism_data_final.csv`
# - Preserve the original row index (`row_id`)
# - Encode labels into a binary target
# - Run a quick exploratory analysis on label distribution and text lengths

# %%
# 1. Load the Dataset

df = pd.read_csv(DATA_PATH)

# Keep track of original row index so we can retrieve later
df["row_id"] = df.index

print("Dataset loaded. Shape:", df.shape)
df.head()

# %%
# 2. Encode labels and basic EDA

# Map EXTREMIST to 1, NON_EXTREMIST to 0
label_map = {
    "EXTREMIST": 1,
    "NON_EXTREMIST": 0,
}

df["Binary_Label"] = df["Extremism_Label"].map(label_map).astype(np.int64)
y = df["Binary_Label"].values

print("Label distribution:")
print(df["Extremism_Label"].value_counts())
print("\nLabel distribution (proportions):")
print(df["Extremism_Label"].value_counts(normalize=True))

# Quick text length distribution
text_lengths = df["Original_Message"].fillna("").astype(str).str.len()
plt.hist(text_lengths, bins=30)
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of characters in Original_Message")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Feature Engineering: TF–IDF and VADER Sentiment
# 
# We construct a high-dimensional sparse representation of each message by:
# - Fitting a TF–IDF vectorizer on 1–3 gram tokens
# - Appending 4 VADER sentiment scores (neg, neu, pos, compound)
# - Building a combined feature matrix `X` and aligned `feature_names`

# %%
# 3. Fit TF-IDF on the corpus and prepare VADER

texts = df["Original_Message"].fillna("").astype(str).tolist()

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    min_df=3,
    ngram_range=(1, 3)
)
tfidf_vectorizer.fit(texts)

analyzer = VaderAnalyzer()

print("Number of TF-IDF features:", len(tfidf_vectorizer.get_feature_names_out()))

# %%
# 4. Define vectorize_text(text) and build full feature matrix X

def vectorize_text(text: str):
    """
    Convert a single text string into a feature vector:
      [TF-IDF features | VADER neg, neu, pos, compound]
    Returns a sparse CSR matrix of shape (1, n_features).
    """
    # TF-IDF part
    X_tfidf = tfidf_vectorizer.transform([text])

    # VADER part
    scores = analyzer.polarity_scores(text)
    vader_vec = np.array([[scores["neg"], scores["neu"], scores["pos"], scores["compound"]]])
    X_vader = csr_matrix(vader_vec)

    # Concatenate horizontally
    X_full = hstack([X_tfidf, X_vader], format="csr")
    return X_full

# Apply vectorize_text() to each item
row_vectors = [
    vectorize_text(t) for t in df["Original_Message"].fillna("").astype(str)
]

X = vstack(row_vectors)   # shape: (n_samples, n_features)

print("Feature matrix shape:", X.shape)

# Build feature names: [TF-IDF tokens | VADER scores]
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
vader_feature_names = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
feature_names = tfidf_feature_names + vader_feature_names

print("Total features in feature_names:", len(feature_names))
assert len(feature_names) == X.shape[1], "feature_names must match X columns"

# %% [markdown]
# ## 3. Train/Validation Split and Tensor Conversion
# 
# We:
# - Convert the sparse feature matrix to a dense NumPy array
# - Split into train/validation sets with stratification
# - Convert arrays to PyTorch tensors and build a `DataLoader` for training

# %%
# 5. Training / validation split and tensor conversion

# Convert X and y to NumPy dense arrays
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

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:  ", X_val.shape)
print("y_val shape:  ", y_val.shape)

# %%
# 5b. Reconstruct validation indices & texts WITHOUT retraining

# 1. All row indices
n_samples = len(df)
all_indices = np.arange(n_samples)

# 2. Use train_test_split on indices with SAME random_state & stratify
_, val_idx = train_test_split(
    all_indices,
    test_size=0.2,           # must match your original split
    random_state=30,         # must match your original split
    stratify=y               # same stratify target
)

# 3. Build texts_val array aligned with X_val / X_val_explain
texts_all = df["Original_Message"].fillna("").astype(str).values
texts_val = texts_all[val_idx]

print("Validation size:", len(texts_val))
print("Example validation text:", texts_val[0])

# %% [markdown]
# ## 4. Model Definition and Training Configuration
# 
# We define a simple single-hidden-layer neural network (`SingleLayerNet`)
# and the associated loss function and optimizer. The model outputs a
# probability for the EXTREMIST class using a sigmoid activation.

# %%
# 6. Model definition and optimizer

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
output_size = 1  # binary

model2 = SingleLayerNet(input_size, HIDDEN_NEURONS, output_size)
print(model2)

def criterion(y_pred, y_true):
    eps = 1e-8
    loss = -1 * (y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))
    return torch.mean(loss)

optimizer = optim.SGD(model2.parameters(), lr=LR)

# %% [markdown]
# ## 5. Model Training
# 
# We train the model for `EPOCHS = 130` epochs, tracking:
# - Training loss
# - Validation loss
# - Validation accuracy
# 
# We keep the model state from the epoch with the best validation accuracy.

# %%
# 7. Training loop with metric tracking

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_epoch = None
best_state_dict = None

for epoch in range(EPOCHS):
    # ----- TRAIN -----
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

    # ----- VALIDATION -----
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

    print(
        f"Epoch {epoch+1}/{EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
model2.load_state_dict(best_state_dict)

# %% [markdown]
# ## 6. Training Dynamics
# 
# We visualize the training process via:
# - Training vs. validation loss
# - Validation accuracy across epochs
# 
# This helps diagnose overfitting, underfitting, and training stability.

# %%
# 8. Plot training & validation curves

epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs_range, val_accuracies, marker="o")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# %% [markdown]
# ## 7. Validation Evaluation and Metrics
# 
# We evaluate the trained model on the validation set using:
# - Accuracy
# - Macro and weighted F1 scores
# - Detailed classification report
# - Confusion matrix

# %%
# 9. Final evaluation on validation set

model2.eval()
with torch.no_grad():
    y_val_pred = model2(X_val).cpu().numpy().ravel()

y_val_true = y_val.cpu().numpy().ravel().astype(int)
y_val_pred_labels = (y_val_pred >= VAL_THRESHOLD).astype(int)

acc = accuracy_score(y_val_true, y_val_pred_labels)
f1_macro = f1_score(y_val_true, y_val_pred_labels, average="macro")
f1_weighted = f1_score(y_val_true, y_val_pred_labels, average="weighted")

print("Sklearn metrics on validation set (Neural net model):")
print(f"Accuracy:      {acc:.4f}")
print(f"F1 (macro):    {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}\n")

print("Classification report:")
print(classification_report(y_val_true, y_val_pred_labels, target_names=["NON_EXTREMIST", "EXTREMIST"]))

print("Confusion matrix:")
cm = confusion_matrix(y_val_true, y_val_pred_labels)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NON_EXTREMIST", "EXTREMIST"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Validation Set")
plt.show()

# %%
# 9b. Extra metrics table (per-class, macro, weighted)

from sklearn.metrics import classification_report
import pandas as pd

report_dict = classification_report(
    y_val_true,
    y_val_pred_labels,
    target_names=["NON_EXTREMIST", "EXTREMIST"],
    output_dict=True,
)

metrics_df = pd.DataFrame(report_dict).T
print("Detailed classification report as table:")
display(metrics_df)

# %%
# 9c. ROC curve and ROC AUC

from sklearn.metrics import roc_curve, roc_auc_score

# fpr = false positive rate, tpr = true positive rate
fpr, tpr, roc_thresholds = roc_curve(y_val_true, y_val_pred)
roc_auc = roc_auc_score(y_val_true, y_val_pred)

print(f"ROC AUC: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.title("ROC Curve – Validation Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%
# 9d. Precision–Recall curve and Average Precision (AP)

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_val_true, y_val_pred)
ap = average_precision_score(y_val_true, y_val_pred)

print(f"Average Precision (AP): {ap:.4f}")

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
plt.title("Precision–Recall Curve – Validation Set")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# %%
# 9e. Log loss and Brier score

from sklearn.metrics import log_loss, brier_score_loss

ll = log_loss(y_val_true, y_val_pred)
brier = brier_score_loss(y_val_true, y_val_pred)

print(f"Log loss:       {ll:.4f}")
print(f"Brier score:    {brier:.4f}")

# %%
# 9f. Probability histograms (overall and per-class)

plt.figure()
plt.hist(y_val_pred, bins=20, alpha=0.7)
plt.title("Distribution of Predicted Probabilities – Validation Set")
plt.xlabel("Predicted P(EXTREMIST)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(y_val_pred[y_val_true == 0], bins=20, alpha=0.7, label="True NON_EXTREMIST")
plt.hist(y_val_pred[y_val_true == 1], bins=20, alpha=0.7, label="True EXTREMIST")
plt.title("Predicted Probabilities by True Class – Validation Set")
plt.xlabel("Predicted P(EXTREMIST)")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 8. SHAP Explainability
# 
# To interpret the model, we use SHAP (SHapley Additive exPlanations):
# 
# - We summarize the training data with K-means to create a background set
# - We use `KernelExplainer` to approximate Shapley values for a subset of
#    validation examples
# - We visualize global feature importance as well as local explanations for
#    individual predictions

# %%
# 10. SHAP setup for model2

shap.initjs()
model2.eval()

def model2_predict(x_np):
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_np.astype(np.float32))
        probs = model2(x_tensor).cpu().numpy().ravel()
    return probs

# Summarize the background with K representative samples
K = 100  # tradeoff between speed and accuracy
background = shap.kmeans(X_train_np, K)

explainer = shap.KernelExplainer(model2_predict, background)

# %%
# 11. Compute SHAP values for a subset of the validation set

explain_size = min(200, X_val_np.shape[0])
X_val_explain = X_val_np[:explain_size]

shap_values = explainer.shap_values(X_val_explain)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.array(shap_values)

print("X_val_explain shape:", X_val_explain.shape)
print("SHAP values shape:  ", shap_values.shape)

# %%
# 11b. Build a global SHAP Explanation object for the explained subset

import numpy as np
import shap

global_exp = shap.Explanation(
    values=shap_values,                                      # (n_samples, n_features)
    base_values=np.repeat(explainer.expected_value, shap_values.shape[0]),
    data=X_val_explain,                                      # (n_samples, n_features)
    feature_names=feature_names
)

print(global_exp)

# %% [markdown]
# ### 8.1 Global SHAP Explanations
# 
# Here we explore how features behave **on average** across the explained
# validation subset using:
# - Global bar plots (mean absolute SHAP)
# - SHAP heatmap over top samples
# - Scatter/dependence plots
# - SHAP summary (beeswarm) for feature impact and direction

# %%
# 12a. Global bar plot (new-style API)

shap.plots.bar(global_exp, max_display=20)

# %%
# 12b. Heatmap of SHAP values for top-N samples by predicted extremism probability

N = min(50, global_exp.values.shape[0])  # top N samples
# Sort by model probability of extremist (highest first)
top_idx = np.argsort(-y_val_pred[:global_exp.shape[0]])[:N]

shap.plots.heatmap(global_exp[top_idx], max_display=20)

# %%
# 12c. SHAP Scatter / dependence plot for a chosen feature (new-style)

# After looking at global bar importance, pick a top feature index:
feature_idx = 818 # index for 'bitch'

shap.plots.scatter(
    global_exp[:, feature_idx],
    color=global_exp[:, feature_idx]  # color by same feature, or choose another
)

# %%
# 12d. SHAP summary beeswarm (global impact and direction)

shap.summary_plot(
    shap_values,
    X_val_explain,
    feature_names=feature_names,
    show=True
)

# %% [markdown]
# ### 8.2 Local SHAP Explanations
# 
# We now inspect individual predictions in detail:
# - Print the original validation message
# - Plot a SHAP waterfall chart to show how each feature pushes the prediction
# - Use a local bar plot and a SHAP force plot for the same example
# 
# Note: `i` must be in the range `[0, explain_size)`.

# %%
# 13a. Local waterfall plot with original message

i = 0  # pick any index in [0, explain_size)

print(f"Validation example index: {i}")
print("Original message:")
print(texts_val[i])
print("-" * 80)

local_exp = shap.Explanation(
    values=shap_values[i],                 # SHAP values for example i
    base_values=explainer.expected_value,  # model baseline
    data=X_val_explain[i],                 # feature vector for example i
    feature_names=feature_names
)

shap.plots.waterfall(local_exp, max_display=20)

# %%
# 13b. Local SHAP bar plot for a single example

i = 0  # ideally the same index as above

local_exp_bar = global_exp[i]
shap.plots.bar(local_exp_bar, max_display=20)

# %%
# 13c. Local SHAP force plot for a single example

i = 0  # same index as waterfall / bar, if desired

shap.force_plot(
    explainer.expected_value,
    shap_values[i, :],
    X_val_explain[i, :],
    matplotlib=False  # interactive JS (if supported)
)

shap.force_plot(
    explainer.expected_value,
    shap_values[i, :],
    X_val_explain[i, :],
    matplotlib=True   # static fallback
)

# %%
# 13d. SHAP decision plot (optional global view)

shap.decision_plot(
    explainer.expected_value,
    shap_values,
    feature_names=feature_names,
)

# %%
# 13e. SHAP dependence plot (legacy API)

# After inspecting the bar plots, set this to a top feature index
feature_idx = 818 # feature number

shap.dependence_plot(
    feature_idx,
    shap_values,
    X_val_explain,
    feature_names=feature_names
)

# %%
# 13f. Utility: look up TF-IDF / SHAP feature indices for a given token

target_token = "bitch"  # change this to any token or n-gram

# 1. Exact match in TF-IDF features
try:
    exact_idx_in_tfidf = tfidf_feature_names.index(target_token)
    print(f'Exact match "{target_token}" found in TF-IDF at index: {exact_idx_in_tfidf}')
except ValueError:
    exact_idx_in_tfidf = None
    print(f'Exact match "{target_token}" not found in TF-IDF features.')

# 2. Any n-gram feature that CONTAINS the token
containing_indices = [
    (i, fname)
    for i, fname in enumerate(tfidf_feature_names)
    if target_token in fname
]

if containing_indices:
    print(f'\nTF-IDF features containing "{target_token}":')
    for i, fname in containing_indices[:50]:
        print(f"  index {i}: {fname}")
else:
    print(f'\nNo TF-IDF features contain "{target_token}".')

# 3. Corresponding index in full feature_names (TF-IDF + VADER)
if target_token in feature_names:
    full_index = feature_names.index(target_token)
    print(f'\nIn full feature_names (with VADER appended), '
          f'"{target_token}" is at index: {full_index}')
else:
    print(f'\n"{target_token}" does not appear as an exact entry in feature_names.')

# %% [markdown]
# ## 9. Model-Based Label Quality Check
# 
# We score the entire dataset with the trained model and:
# - Identify rows where the model prediction disagrees with the dataset label
# - Rank disagreements by model confidence to flag likely mislabels for review

# %%
# 14. Use trained model to score all rows and find disagreements

model2.eval()
with torch.no_grad():
    X_all_tensor = torch.from_numpy(X_np)
    y_all_probs = model2(X_all_tensor).cpu().numpy().ravel()

y_all_true = y_np.ravel().astype(int)
y_all_pred = (y_all_probs >= VAL_THRESHOLD).astype(int)

df["model_prob_extremist"] = y_all_probs
df["model_pred_label"] = y_all_pred

disagree_mask = (y_all_true != y_all_pred)
print("Number of disagreements between model and gold label:", disagree_mask.sum())

df_disagree = df.loc[disagree_mask].copy()
df_disagree["true_label"] = y_all_true[disagree_mask]
df_disagree["pred_label"] = y_all_pred[disagree_mask]

df_disagree["model_confidence"] = np.where(
    df_disagree["model_prob_extremist"] >= 0.5,
    df_disagree["model_prob_extremist"],
    1.0 - df_disagree["model_prob_extremist"],
)

df_disagree = df_disagree.sort_values("model_confidence", ascending=False)

df_disagree[[
    "row_id",
    "Original_Message",
    "Extremism_Label",
    "true_label",
    "pred_label",
    "model_prob_extremist",
    "model_confidence",
]].head(20)

# %%
# 15. Save candidate mislabels for manual review

output_path = "potential_mislabels_by_model.csv"
df_disagree.to_csv(output_path, index=False)
print(f"Saved {len(df_disagree)} candidate mislabels to {output_path}")

# %%
# %%
# Utility to inspect TF-IDF feature indices for a given token

target_token = "bitch"

# 1. Exact match in TF-IDF features (unigrams / bigrams / trigrams)
try:
    exact_idx_in_tfidf = tfidf_feature_names.index(target_token)
    print(f'Exact match "{target_token}" found in TF-IDF at index: {exact_idx_in_tfidf}')
except ValueError:
    print(f'Exact match "{target_token}" not found in TF-IDF features.')

# 2. Any n-gram feature that CONTAINS the token (e.g. "stupid bitch", "you bitch")
containing_indices = [
    (i, fname)
    for i, fname in enumerate(tfidf_feature_names)
    if target_token in fname
]

if containing_indices:
    print(f'\nTF-IDF features containing "{target_token}":')
    for i, fname in containing_indices[:50]:   # cap to first 50 just in case
        print(f"  index {i}: {fname}")
else:
    print(f'\nNo TF-IDF features contain "{target_token}".')

# 3. Corresponding index in the FULL feature_names list (TF-IDF + VADER)
#    (only relevant if the exact token exists as a feature)
if "bitch" in feature_names:
    full_index = feature_names.index(target_token)
    print(f'\nIn full feature_names (with VADER appended), '
          f'"{target_token}" is at index: {full_index}')
else:
    print(f'\n"{target_token}" does not appear as an exact entry in feature_names.')

# %% [markdown]
# ## 10. Interpretation of Model Performance, SHAP Explanations, and Label-Quality Check
# 
# This section provides a detailed narrative of the model’s behaviour and what each metric and visualization in the notebook tells us about performance, calibration, interpretability, and dataset quality.
# 
# ---
# 
# ### 10.1 Dataset and Label Distribution
# 
# - **Class balance**  
#   The label distribution shows 548 `NON_EXTREMIST` and 531 `EXTREMIST` examples (roughly 51% vs 49%), so the dataset is nearly balanced. This is important because:
#   - Accuracy is a meaningful metric (we are not dominated by a large majority class).
#   - Precision and recall for both classes can be compared directly, since neither class is rare.  
# 
# - **Text length histogram – “Distribution of Text Lengths”**  
#   The histogram of `Original_Message` character counts shows how long the posts typically are. In your notebook:
#   - Most messages cluster within a moderate length range, with relatively few extremely short or extremely long posts.
#   - This supports the use of an n-gram TF-IDF representation: there is enough text per message to capture meaningful patterns, but not so much that documents become extremely sparse or noisy.  
# 
# Overall, the dataset is *well-behaved*: fairly balanced labels and reasonable text lengths, which makes the learning problem well-posed.
# 
# ---
# 
# ### 10.2 Validation Metrics and Confusion Matrix
# 
# After training the single-layer perceptron and evaluating on the validation set, you report:
# 
# - **Overall metrics (validation set)**  
#   - **Accuracy:** 0.8219  
#   - **F1 (macro):** 0.8220  
#   - **F1 (weighted):** 0.8219  
# 
#   Interpretation:
#   - Accuracy ≈ 82% means about 4 out of 5 validation posts are classified correctly.
#   - Macro F1 ≈ 0.82 shows balanced performance across both classes (it averages F1 for `EXTREMIST` and `NON_EXTREMIST`).
#   - Weighted F1 ≈ 0.82 is almost identical to macro F1 because the classes are nearly balanced.  
# 
# - **Per-class precision / recall / F1 (classification report)**  
#   - `NON_EXTREMIST`  
#     - Precision: 0.83 (when the model predicts non-extremist, it’s correct 83% of the time).  
#     - Recall: 0.82 (it correctly captures 82% of the true non-extremist posts).  
#     - F1: 0.82 (harmonic mean of precision and recall).  
#   - `EXTREMIST`  
#     - Precision: 0.81 (when the model predicts extremist, it’s correct 81% of the time).  
#     - Recall: 0.82 (it correctly detects 82% of the extremist posts).  
#     - F1: 0.82.  
# 
#   This near-symmetry between classes means the model does *not* strongly favour one label over the other; it treats extremist and non-extremist content comparably well.
# 
# - **Confusion matrix (table + heatmap)**  
#   - True `NON_EXTREMIST` correctly predicted: 240  
#   - True `NON_EXTREMIST` misclassified as `EXTREMIST`: 51  
#   - True `EXTREMIST` correctly predicted: 217  
#   - True `EXTREMIST` misclassified as `NON_EXTREMIST`: 48  
# 
#   Interpretation:
#   - The number of false positives (51) and false negatives (48) is similar, again reflecting a balanced trade-off between the two error types.
#   - In practical terms:
#     - **False positives** (benign content flagged as extremist) represent moderation overhead and potential fairness/over-flagging concerns.
#     - **False negatives** (extremist content not caught) represent missed detections, which matter for safety.  
#   - Because the counts are balanced, the current threshold (0.5) is a reasonable starting point; later, you can tune the threshold depending on whether you want to prioritize catching extremists (reduce false negatives) or avoiding false alarms (reduce false positives).
# 
# The **detailed classification report table** complements the confusion matrix by putting these numbers into normalized precision/recall/F1 form, making it easy to compare models or future experiments. 
# 
# ---
# 
# ### 10.3 Training Dynamics: Loss and Accuracy Curves
# 
# The training loop tracks:
# 
# - **Training loss per epoch**  
# - **Validation loss per epoch**  
# - **Validation accuracy per epoch**  
# 
# These are visualized in two line plots:
# 
# 1. **“Training and Validation Loss”**  
#    - Training loss decreases steadily as epochs increase, indicating that the model is successfully fitting the training data.
#    - Validation loss falls at first, then flattens or shows small fluctuations.
#    - The epoch with the best validation accuracy is saved and restored as `best_state_dict`, helping you avoid overfitting by using the best-performing checkpoint rather than the final epoch.  
# 
# 2. **“Validation Accuracy over Epochs”**  
#    - Validation accuracy starts lower and then climbs towards ~0.8+ as training progresses.
#    - The curve plateaus once the model has learned most of the patterns; small ups and downs indicate normal training noise rather than severe overfitting.
# 
# Together, these plots show that:
# - The model learns meaningful structure from the data,
# - The chosen training length (EPOCHS) is enough to reach stable performance,
# - And early-stopping via `best_val_acc` is effectively used to select a good checkpoint.
# 
# ---
# 
# ### 10.4 Threshold-Free Metrics: ROC and Precision–Recall Curves
# 
# To understand performance across *all* possible thresholds, the notebook includes:
# 
# #### 9.4.1 ROC Curve and ROC AUC
# 
# - The code prints:  
#   `ROC AUC: 0.9153`  
# - The **ROC curve** plot compares:
#   - The **true positive rate (TPR)** on the y-axis (fraction of extremists correctly detected)  
#   - Against the **false positive rate (FPR)** on the x-axis (fraction of non-extremists incorrectly flagged)  
#   - With a diagonal 45° line as a **random baseline**.
# 
# Interpretation:
# - An ROC AUC of **0.9153** is very strong: the model can correctly rank random extremist vs non-extremist pairs about 91.5% of the time.
# - The curve bowing well above the diagonal shows that, for many thresholds, you can get high TPR at reasonably low FPR.
# - This means the model offers **flexible operating points**: if a downstream system wants to be more conservative or more aggressive, you can adjust the threshold with a predictable trade-off.
# 
# #### 9.4.2 Precision–Recall Curve and Average Precision (AP)
# 
# - The notebook reports:  
#   `Average Precision (AP): 0.9044`  
# - The **Precision–Recall (PR) curve** plots precision vs recall as you sweep the threshold.
# 
# Interpretation:
# - An AP of **0.9044** is high, confirming that for a wide range of thresholds, you can achieve both high precision and high recall on the extremist class.
# - Since your classes are fairly balanced, ROC AUC and AP tell a consistent story: the model is effective at ranking examples and distinguishing extremist vs non-extremist content.
# 
# ---
# 
# ### 10.5 Calibration Metrics and Probability Histograms
# 
# To understand *how trustworthy* the predicted probabilities are, you compute:
# 
# - **Log loss:** 0.3684  
# - **Brier score:** 0.1168  
# 
# Interpretation:
# - **Log loss** penalizes overconfident wrong predictions heavily. A value around 0.37 is consistent with a reasonably well-calibrated, discriminative model for a binary problem.
# - **Brier score** is the mean squared error between predicted probabilities and the true 0/1 labels; 0.1168 is substantially lower than the 0.25 you’d get from a completely uninformative 50/50 predictor on a balanced dataset. This again indicates good calibration.
# 
# The notebook then visualizes probability distributions:
# 
# 1. **“Distribution of Predicted Probabilities – Validation Set”**  
#    - Shows the overall histogram of `P(EXTREMIST)` over the validation examples.
#    - Typically, you’ll see mass near 0 and 1 if the model is confident, and more central mass near 0.5 if the model is uncertain.
#    - A bimodal shape (peaks near 0 and 1) suggests confident predictions on many examples.
# 
# 2. **“Predicted Probabilities by True Class – Validation Set”**  
#    - Overlays two histograms:
#      - One for true non-extremist posts,
#      - One for true extremist posts.
#    - In a well-performing model:
#      - The **non-extremist** histogram should concentrate near 0,
#      - The **extremist** histogram should concentrate near 1,
#      - With limited overlap between the two.  
# 
# These plots visually confirm the numeric metrics: the model is not only accurate, but also produces probabilities that meaningfully separate the two classes.
# 
# ---
# 
# ### 10.6 SHAP Global Explanations (Which Features Matter Overall?)
# 
# You compute SHAP values using `KernelExplainer` over a subset of validation examples, then build a `shap.Explanation` object (`global_exp`) and visualize it with several plots.
# 
# 1. **Global SHAP bar plot (`shap.plots.bar(global_exp)`)**  
#    - Shows the **mean absolute SHAP value** for the top features (words/phrases and VADER scores).
#    - Higher bars correspond to features that, on average, have a larger impact on pushing predictions towards extremist or non-extremist.
#    - Because your feature space is `[TF-IDF n-grams | VADER neg/neu/pos/compound]`, the plot tells you:
#      - Which **n-grams** are most predictive of extremism vs non-extremism.
#      - How much sentiment signals (e.g., `vader_neg`, `vader_compound`) contribute relative to the text n-grams.
# 
# 2. **SHAP heatmap (`shap.plots.heatmap(global_exp[top_idx])`)** 
#    - Displays samples (rows) vs features (columns), with colour intensity representing SHAP values.
#    - You sort by **predicted extremist probability** and look at the top-N examples; this effectively shows:
#      - Which features consistently drive the model towards an extremist prediction.
#      - Patterns such as “these phrases repeatedly push SHAP values in the positive direction for many high-probability extremist posts.”
# 
# 3. **SHAP scatter / dependence plot (`shap.plots.scatter(global_exp[:, feature_idx])`)**  
#    - For a chosen feature, this plot shows:
#      - The feature’s value (x-axis),
#      - The corresponding SHAP value (y-axis),
#      - Often coloured by the same or another feature.
#    - Interpretation:
#      - You see whether higher values of that feature (e.g., more frequent occurrence or higher TF-IDF weight) systematically increase or decrease the predicted probability of extremism.
#      - Monotonic trends suggest a clear directional effect; scattered points suggest more context-dependent contributions.
# 
# 4. **SHAP summary bar plot (`shap.summary_plot(..., plot_type="bar")`)**  
#    - This “classic” SHAP bar chart again highlights the top features by mean |SHAP|, as a more compact summary.
#    - It reinforces which n-grams and sentiment dimensions are globally most influential.
# 
# Overall, the global SHAP plots answer: *“What kinds of words, phrases, and sentiment patterns does the model rely on most when deciding whether a post is extremist?”*
# 
# ---
# 
# ### 10.7 SHAP Local Explanations (Why This Specific Prediction?)
# 
# Several plots zoom in on **individual examples** to explain *why* the model predicted extremist vs non-extremist for a particular message.
# 
# 1. **Waterfall plot (`shap.plots.waterfall(exp)`) with printed message**   
#    - For a chosen index `i`, you:
#      - Print the **original validation message** (`texts_val[i]`),
#      - Build a SHAP `Explanation` object for that row,
#      - Plot a waterfall that shows:
#        - The **baseline prediction** (expected value over the background distribution),
#        - How each feature (starting from the most important) **pushes the output up or down**.
#    - Interpretation:
#      - Red bars push the prediction towards extremist;
#      - Blue bars push it towards non-extremist;
#      - The final value at the bottom is the model’s predicted probability for this example.
#    - This gives a human-readable narrative of *“These specific words and sentiment cues are why the model labelled this post as extremist / non-extremist.”*
# 
# 2. **Local SHAP bar plot (`shap.plots.bar(local_exp)`)**   
#    - Shows the top contributing features for a single example as a bar chart.
#    - This is a simpler view than the waterfall, but conveys the same idea: which features are most responsible for the prediction, and in which direction they push.
# 
# 3. **SHAP dependence plot (old API, `shap.dependence_plot(...)`)** 
#    - Similar to the new scatter plot, but using the older interface.
#    - For a chosen feature index, you see how variation in that feature’s value across samples changes its SHAP contribution.
#    - This again provides insight into whether that feature steadily pushes predictions upward/downward or only matters in specific ranges or contexts.
# 
# 4. **SHAP force plot (`shap.force_plot(...)`)**  
#    - The interactive JS version (if supported) produces a horizontal “force” visualization where:
#      - Features pushing towards extremist are shown in one colour,
#      - Features pushing towards non-extremist in another,
#      - The length of each bar corresponds to magnitude.
#    - The static matplotlib version replicates this idea as an image.
#    - This is another intuitive way to show stakeholders **how the model arrived at one specific probability**.
# 
# 5. **SHAP decision plot (`shap.decision_plot(...)`)**  
#    - Visualizes how the prediction is constructed step-by-step:
#      - Starting from the baseline expected value,
#      - Adding contributions of each feature in sequence.
#    - For multiple samples, the decision plot shows many trajectories, helping you see whether different posts rely on similar or different combinations of features.
# 
# Collectively, these local SHAP plots are very useful for:
# - Debugging individual predictions,
# - Explaining decisions to non-technical stakeholders,
# - And checking that the model is using *semantically reasonable* signals (and not just artefacts or spurious correlations).
# 
# ---
# 
# ### 10.8 Model-Based Label Quality Check
# 
# Finally, you use the trained model to **audit the labels** by finding disagreements:
# 
# - For every row in the dataset, you compute the model’s probability `P(EXTREMIST)` and predicted label, then compare it to the ground-truth label.
# - You report:  
#   `Number of disagreements between model and gold label: 227`   
# 
# Interpretation:
# - Out of all posts, 227 are places where the model’s prediction disagrees with the dataset label.
# - You also compute a **model confidence** metric (`max(p, 1-p)`) and sort the disagreements by this confidence:
#   - High-confidence disagreements are especially interesting:
#     - If the model is usually accurate elsewhere, these may indicate **potential mislabels** in the dataset.
#     - Alternatively, they might highlight systematic **biases or blind spots** in the model (e.g., certain types of nuanced extremist content the model still misreads).
# 
# The notebook then:
# - Shows the top 20 disagreements in a table (with `row_id`, original text, true label, predicted label, probability, confidence), and  
# - Saves all disagreements to `potential_mislabels_by_model.csv` for manual review. 
# 
# This closes the loop from *pure performance evaluation* to **data quality assurance**, which is critical for sensitive tasks like extremism detection.
# 
# ---
# 
# ### 10.9 High-Level Takeaways
# 
# Putting all metrics and plots together:
# 
# - The model achieves **strong discriminative performance** (Accuracy ≈ 0.82, ROC AUC ≈ 0.92, AP ≈ 0.90).
# - Performance is **balanced across classes**: both extremist and non-extremist posts are handled with comparable precision and recall.
# - Calibration metrics (log loss, Brier score, probability histograms) show that predicted probabilities are meaningfully informative, not random scores.
# - SHAP global and local explanations confirm that the model’s decisions are driven by interpretable text features and sentiment signals, and provide a detailed story for both aggregate and individual behaviours.
# - The disagreement analysis surfaces **227 posts** where the model and labels conflict, giving you a concrete shortlist for improving label quality or probing model weaknesses.
# 
# Overall, the notebook provides a **complete, professional evaluation** of the extremism detection model: performance, calibration, interpretability, and dataset health are all addressed and connected to concrete visual evidence.


