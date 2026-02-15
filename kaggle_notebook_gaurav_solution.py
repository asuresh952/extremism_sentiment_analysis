# Extremism Detection using Hybrid Ensemble
# Reproducing Gaurav's Methodology on a Single Dataset

# %% [markdown]
## 📋 Overview
# 
# This notebook implements a **hybrid ensemble approach** for detecting extremist content in social media posts.
# 
# **Methodology (based on 0.976 scoring approach):**
# - **Data Cleaning**: Correcting noisy labels
# - **Model A**: TF-IDF + Calibrated SVM (Traditional ML)
# - **Model B**: RoBERTa-Base Fine-tuning with 3-seed ensemble (Deep Learning)
# - **Hybrid Ensemble**: 60% RoBERTa + 40% SVM weighted combination
# 
# **Data Flow:**
# 1. Load single dataset
# 2. Clean and standardize labels
# 3. Split into train/test (80/20)
# 4. Train both models
# 5. Evaluate ensemble performance

# %% [markdown]
## 📦 Installation & Imports

# %%
# Install required packages
!pip install -q transformers datasets accelerate

# %%
# Core imports
import pandas as pd
import numpy as np
import torch
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
## ⚙️ Configuration
# 
# **Model Settings:**
# - Base Model: `roberta-base` (125M parameters)
# - Batch Size: 16 (adjust based on GPU memory)
# - Epochs: 3 (optimal balance)
# - Seeds: [42, 2024, 999] for ensemble stability
# - Test Split: 20% held out for evaluation
# 
# **Why 3 seeds?**
# - Reduces variance in predictions
# - Creates a bagging-like effect
# - Improves generalization

# %%
# Configuration parameters
MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
EPOCHS = 3
SEEDS = [42, 2024, 999]  # Multi-seed ensemble
TEST_SIZE = 0.2          # 80/20 train/test split
RANDOM_STATE = 42

print(f"Model: {MODEL_NAME}")
print(f"Training with {len(SEEDS)} seeds for ensemble robustness")
print(f"Test set size: {TEST_SIZE*100}%")

# %% [markdown]
## 📂 Data Loading
# 
# **Dataset Source**: Single curated extremism detection dataset
# 
# **Expected columns** (will be standardized):
# - Text column: `Original_Message`, `message`, or `text`
# - Label column: `Extremism_Label` or `label`
# 
# **Note**: Labels may be in various formats (text/numeric) and will be cleaned

# %%
print("⏳ Loading Dataset...")

# Load the single dataset
df = pd.read_csv('/kaggle/input/digital-extremism-detection-curated-dataset/extremism_data_final.csv', 
                 on_bad_lines='skip')

print(f"✓ Dataset loaded: {len(df)} samples")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(3))

# %% [markdown]
## 🔧 Data Standardization
# 
# **Problem**: Different column naming conventions
# 
# **Solution**: Standardize to `text` and `label` columns

# %%
def standardize_df(df):
    """
    Standardizes column names across different datasets.
    Handles various naming conventions for text and label columns.
    """
    df = df.copy()
    
    # Standardize text column
    if 'Original_Message' in df.columns:
        df['text'] = df['Original_Message']
    elif 'message' in df.columns:
        df['text'] = df['message']
    elif 'Original_Message,Extremism_Label' in df.columns:
        df['text'] = df['Original_Message,Extremism_Label']
    else:
        df['text'] = df.iloc[:, 0]
    
    # Standardize label column
    if 'Extremism_Label' in df.columns:
        df['label'] = df['Extremism_Label']
    elif 'label' in df.columns:
        df['label'] = df['label']
    else:
        if df.shape[1] > 1:
            df['label'] = df.iloc[:, 1]
    
    return df

# Apply standardization
df_clean = standardize_df(df)

print("✓ Data standardization complete")
print(f"Standardized columns: {df_clean[['text', 'label']].columns.tolist()}")
print(f"\nSample labels (before cleaning):")
print(df_clean['label'].value_counts().head(10))

# %% [markdown]
## 🧹 Label Cleaning Pipeline
# 
# **Problem**: Inconsistent label formats in the dataset:
# - Text labels: "extremist", "non-extremist", "hate", "safe"
# - Numeric labels: 0, 1, "0.0", "1.0"
# - Mixed case and whitespace
# 
# **Solution**:
# 1. Convert to lowercase and strip whitespace
# 2. Map text variations to binary (0/1)
# 3. Handle edge cases
# 4. Remove duplicates and missing values

# %%
def clean_label(val):
    """
    Cleans and standardizes label values.
    
    Returns:
        0 for non-extremist content
        1 for extremist content
    """
    val = str(val).lower().strip()
    
    # Non-extremist patterns
    if 'non' in val or 'not' in val or 'normal' in val or 'safe' in val:
        return 0
    
    # Extremist patterns
    if 'extremist' in val or 'hate' in val or 'offensive' in val:
        return 1
    
    # Numeric values
    if val in ['0', '0.0']:
        return 0
    if val in ['1', '1.0']:
        return 1
    
    # Default to non-extremist for safety
    return 0

# Apply label cleaning
df_clean['label'] = df_clean['label'].apply(clean_label)
df_clean['text'] = df_clean['text'].astype(str).fillna("")

# Remove duplicates and missing values
initial_size = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['text'])
df_clean = df_clean.dropna(subset=['label'])
df_clean['label'] = df_clean['label'].astype(int)

print(f"✓ Removed {initial_size - len(df_clean)} duplicates/invalid entries")
print(f"Final dataset size: {len(df_clean)} samples")
print(f"\nLabel distribution:")
print(df_clean['label'].value_counts())
print(f"Balance: {df_clean['label'].value_counts(normalize=True).round(3).to_dict()}")

# %% [markdown]
## 🔀 Train/Test Split
# 
# **Strategy**: 80/20 stratified split
# - Ensures both classes are proportionally represented
# - Test set will be used for final evaluation
# - No data leakage between sets

# %%
print(f"\n⏳ Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")

# Stratified split to maintain class balance
train_df, test_df = train_test_split(
    df_clean,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_clean['label']
)

print(f"✓ Train set: {len(train_df)} samples")
print(f"✓ Test set: {len(test_df)} samples")
print(f"\nTrain label distribution:")
print(train_df['label'].value_counts())
print(f"\nTest label distribution:")
print(test_df['label'].value_counts())

# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# %% [markdown]
## 🤖 Model A: TF-IDF + Calibrated SVM
# 
# **Why SVM?**
# - Fast training and inference
# - Excellent with TF-IDF features
# - Provides complementary signal to neural models
# 
# **Key Features**:
# - N-grams (1-3): Captures phrases like "kill them all"
# - Calibration: Converts SVM scores to proper probabilities
# - Class balancing: Handles imbalanced datasets

# %%
print("\n🔹 Training TF-IDF + SVM Model...")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=2,            # Ignore very rare terms
    sublinear_tf=True    # Use log scaling for term frequency
)

X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])

print(f"TF-IDF feature dimension: {X_train_tfidf.shape[1]}")

# Train calibrated SVM
base_svc = LinearSVC(
    class_weight='balanced',  # Handle class imbalance
    C=1.0,                    # Regularization
    random_state=42,
    dual=False                # Primal optimization
)

model_tfidf = CalibratedClassifierCV(base_svc, cv=5)  # 5-fold calibration
model_tfidf.fit(X_train_tfidf, train_df['label'])

# Generate probabilities
probs_tfidf_test = model_tfidf.predict_proba(X_test_tfidf)[:, 1]

print(f"✓ SVM training complete")
print(f"Probability range: [{probs_tfidf_test.min():.3f}, {probs_tfidf_test.max():.3f}]")

# Quick evaluation
preds_tfidf = (probs_tfidf_test >= 0.5).astype(int)
acc_tfidf = accuracy_score(test_df['label'], preds_tfidf)
print(f"SVM accuracy on test set: {acc_tfidf:.4f}")

# %% [markdown]
## 🧠 Model B: RoBERTa Ensemble (3 Seeds)
# 
# **Architecture**: RoBERTa-Base
# - 125M parameters
# - Pre-trained on 160GB of text
# - Superior to BERT on most NLP tasks
# 
# **Ensemble Strategy**:
# - Train 3 independent models with different random seeds
# - Average their probability predictions
# - Reduces variance and improves stability
# 
# **Training Details**:
# - Learning rate: 2e-5
# - Max sequence length: 128 tokens
# - FP16 mixed precision training

# %%
print(f"\n🔹 Starting RoBERTa Ensemble (Seeds: {SEEDS})...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    """Tokenize text data with padding and truncation"""
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# Prepare datasets
ds_train = Dataset.from_pandas(train_df[['text', 'label']])
ds_test = Dataset.from_pandas(test_df[['text', 'label']])

ds_train = ds_train.map(tokenize_fn, batched=True)
ds_test = ds_test.map(tokenize_fn, batched=True)

print(f"✓ Tokenization complete")
print(f"Training samples: {len(ds_train)}")
print(f"Test samples: {len(ds_test)}")

# %% [markdown]
## 🔄 Training Loop (3 Seeds)
# 
# **Process per seed**:
# 1. Initialize model with seed
# 2. Train for 3 epochs
# 3. Generate predictions on test set
# 4. Store probabilities
# 5. Clean up GPU memory
# 
# **Expected time**: ~15-20 minutes per seed on T4 GPU

# %%
all_seed_probs = []

for i, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*60}")
    print(f"Training Seed {seed} ({i}/{len(SEEDS)})")
    print(f"{'='*60}")
    
    # Clean up previous model
    if os.path.exists("/kaggle/working/checkpoints"):
        shutil.rmtree("/kaggle/working/checkpoints")
    torch.cuda.empty_cache()
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )
    
    # Training configuration
    args = TrainingArguments(
        output_dir="/kaggle/working/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_strategy="no",  # Don't save checkpoints
        report_to="none",    # Disable logging
        fp16=True,           # Mixed precision
        seed=seed,
        data_seed=seed
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        processing_class=tokenizer
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Generate predictions
    print("Generating predictions...")
    preds = trainer.predict(ds_test)
    probs = torch.nn.functional.softmax(
        torch.tensor(preds.predictions), 
        dim=1
    )[:, 1].numpy()
    
    all_seed_probs.append(probs)
    
    print(f"✓ Seed {seed} complete")
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

# Average predictions from all seeds
avg_roberta_probs = np.mean(all_seed_probs, axis=0)
print(f"\n✓ RoBERTa ensemble complete")
print(f"Final probability range: [{avg_roberta_probs.min():.3f}, {avg_roberta_probs.max():.3f}]")

# Quick evaluation
preds_roberta = (avg_roberta_probs >= 0.5).astype(int)
acc_roberta = accuracy_score(test_df['label'], preds_roberta)
print(f"RoBERTa accuracy on test set: {acc_roberta:.4f}")

# %% [markdown]
## 🎯 Hybrid Ensemble
# 
# **Ensemble Weights**:
# - 60% RoBERTa (better at context understanding)
# - 40% SVM (better at keyword patterns)
# 
# **Prediction Strategy**:
# - Weighted average of probabilities
# - Threshold at 0.5

# %%
print("\n🔹 Creating hybrid ensemble predictions...")

# Weighted ensemble: 60% RoBERTa + 40% SVM
ensemble_probs = (0.6 * avg_roberta_probs) + (0.4 * probs_tfidf_test)

# Make predictions
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

print(f"✓ Ensemble predictions complete")
print(f"Probability range: [{ensemble_probs.min():.3f}, {ensemble_probs.max():.3f}]")

# %% [markdown]
## 📊 Model Evaluation
# 
# **Metrics**:
# - Accuracy: Overall correctness
# - Precision: Of predicted extremist, how many are correct
# - Recall: Of actual extremist, how many did we catch
# - F1-Score: Harmonic mean of precision and recall

# %%
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

# Calculate metrics for each model
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    'SVM (TF-IDF)': preds_tfidf,
    'RoBERTa (3-seed ensemble)': preds_roberta,
    'Hybrid Ensemble (60/40)': ensemble_preds
}

results = []
for name, preds in models.items():
    acc = accuracy_score(test_df['label'], preds)
    prec = precision_score(test_df['label'], preds)
    rec = recall_score(test_df['label'], preds)
    f1 = f1_score(test_df['label'], preds)
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc:.4f}",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}",
        'F1-Score': f"{f1:.4f}"
    })
    
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# %% [markdown]
## 📈 Detailed Analysis: Hybrid Ensemble

# %%
print("\n" + "="*60)
print("HYBRID ENSEMBLE - DETAILED RESULTS")
print("="*60)

# Accuracy
ensemble_acc = accuracy_score(test_df['label'], ensemble_preds)
print(f"\nAccuracy: {ensemble_acc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(test_df['label'], ensemble_preds, 
                          target_names=['Non-Extremist', 'Extremist']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(test_df['label'], ensemble_preds)
print(pd.DataFrame(cm, 
                   columns=['Predicted Non-Extremist', 'Predicted Extremist'],
                   index=['Actual Non-Extremist', 'Actual Extremist']))

# %% [markdown]
## 💾 Save Results
# 
# **Outputs**:
# - Test predictions with probabilities
# - Model comparison metrics

# %%
# Save test predictions
test_results = pd.DataFrame({
    'text': test_df['text'],
    'true_label': test_df['label'],
    'svm_prob': probs_tfidf_test,
    'roberta_prob': avg_roberta_probs,
    'ensemble_prob': ensemble_probs,
    'prediction': ensemble_preds
})

test_results.to_csv('test_predictions.csv', index=False)
print("✓ Test predictions saved to 'test_predictions.csv'")

# Save model comparison
results_df.to_csv('model_comparison.csv', index=False)
print("✓ Model comparison saved to 'model_comparison.csv'")

print(f"\nFirst 10 predictions:")
print(test_results[['text', 'true_label', 'ensemble_prob', 'prediction']].head(10))

# %% [markdown]
## 🎯 Summary
# 
# **What we did**:
# 1. ✓ Loaded and cleaned a single extremism detection dataset
# 2. ✓ Split into train (80%) and test (20%) sets
# 3. ✓ Trained TF-IDF + Calibrated SVM
# 4. ✓ Fine-tuned RoBERTa with 3-seed ensemble
# 5. ✓ Combined models with 60/40 weighted ensemble
# 6. ✓ Evaluated all models on held-out test set
# 
# **Key Methodology** (from 0.976-scoring approach):
# - Robust label cleaning
# - Seed ensemble for stability
# - Hybrid approach combining traditional ML and deep learning
# 
# The hybrid ensemble typically outperforms individual models! 🚀