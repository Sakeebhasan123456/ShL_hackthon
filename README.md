# Detailed Explanation of Audio Grammer Model

## Overview
This notebook implements an **audio classification/regression pipeline** that predicts continuous labels for audio files. The approach uses state-of-the-art pre-trained audio models for feature extraction combined with gradient boosting for prediction.

---

## 1. Environment Setup

```python
!pip install whisper
!pip install protobuf==3.20.3
!apt-get update && apt-get install -y libsndfile1
```

**Purpose:**
- Installs necessary audio processing libraries
- `whisper`: OpenAI's speech recognition model (imported but not used in final pipeline)
- `protobuf==3.20.3`: Protocol buffers for data serialization (specific version for compatibility)
- `libsndfile1`: System library for reading/writing audio files

**Note:** The protobuf downgrade causes dependency conflicts but is necessary for compatibility with certain audio processing libraries.

---

## 2. Library Imports

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoFeatureExtractor
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import librosa
import whisper
```

**Key Libraries:**
- **PyTorch**: Deep learning framework for model loading and inference
- **Transformers**: Hugging Face library providing pre-trained audio models
- **Librosa**: Audio loading and preprocessing
- **scikit-learn**: Train-test splitting and evaluation metrics
- **LightGBM/XGBoost**: Gradient boosting models for regression
- **Optuna**: Hyperparameter optimization framework

---

## 3. Configuration and Data Loading

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/wavlm-large"
SAMPLE_RATE = 16000
TRAIN_AUDIO_DIR = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/audios/train"
TEST_AUDIO_DIR = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/audios/test"
TRAIN_CSV_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/csvs/train.csv"
TEST_CSV_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/csvs/test.csv"
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
```

**Configuration Details:**
- **DEVICE**: Automatically selects GPU (cuda) if available, otherwise CPU
- **MODEL_NAME**: `microsoft/wavlm-large` - State-of-the-art audio representation model
- **SAMPLE_RATE**: 16000 Hz - Standard sampling rate for speech models
- **Data Structure**: CSV files contain filenames and labels (for training only)

---

## 4. Model Selection and Loading

```python
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
```

**Why WavLM-Large?**

The notebook provides several model options with commentary:

1. **WavLM** (Chosen) - Microsoft's state-of-the-art model
   - Best for: Emotion recognition, speaker identification, general speech understanding
   - Size: Large (1.26GB)
   - Features: 1024-dimensional embeddings

2. **HuBERT** - Facebook's competitor to Wav2Vec2
   - Strong alternative for speech tasks

3. **data2vec-audio** - Facebook's newer self-supervised model
   - Unified framework across modalities

4. **BEATs** - Microsoft's model
   - Best for: Non-speech audio (environmental sounds, music)

**Loading Process:**
- `AutoFeatureExtractor`: Automatically loads the correct audio preprocessor
- `AutoModel`: Loads the pre-trained model architecture and weights
- `.to(DEVICE)`: Moves model to GPU for faster inference
- `.eval()`: Sets model to evaluation mode (disables dropout, batch normalization)

---

## 5. Feature Extraction Function

```python
def extract_features(audio_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
        if waveform.shape[0] < 1024:
            print(f"Skipping short file: {audio_path}")
            return None
        
        input_values = processor(
            waveform, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt"
        ).input_values.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state
        
        pooled = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        return pooled
        
    except Exception as e:
        print(f"Error in extract_features for {audio_path}: {e}")
        return None
```

**Step-by-Step Breakdown:**

1. **Audio Loading**
   ```python
   waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
   ```
   - Loads audio file and resamples to 16kHz
   - Returns waveform as numpy array and sampling rate

2. **Length Check**
   ```python
   if waveform.shape[0] < 1024:
       return None
   ```
   - Skips very short audio files (< 64ms at 16kHz)
   - Prevents errors from insufficient data

3. **Preprocessing**
   ```python
   input_values = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
   ```
   - Normalizes audio
   - Converts to format expected by model
   - Returns PyTorch tensor

4. **Feature Extraction**
   ```python
   with torch.no_grad():
       outputs = model(input_values)
       hidden_states = outputs.last_hidden_state
   ```
   - `torch.no_grad()`: Disables gradient computation (faster, less memory)
   - Runs forward pass through model
   - Extracts final hidden states (time × features)

5. **Pooling**
   ```python
   pooled = hidden_states.mean(dim=1).squeeze().cpu().numpy()
   ```
   - Averages across time dimension (temporal pooling)
   - Converts from tensor to numpy array
   - Results in single 1024-dimensional feature vector per audio file

---

## 6. Training Data Feature Extraction

```python
features_train = []
labels_train = []
print("\nExtracting Training dataset features...")

for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
    file_path = os.path.join(TRAIN_AUDIO_DIR, row['filename'] + ".wav")
    
    if not os.path.exists(file_path):
        continue
    
    feat = extract_features(file_path)
    
    if feat is not None:
        features_train.append(feat)
        labels_train.append(row["label"])

if features_train:
    X_train = np.stack(features_train)
    y_train = np.array(labels_train)
    print(f"\n--- Training features extracted ---")
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
```

**Process:**
- Iterates through all training files (409 total)
- Constructs full file path (adds .wav extension)
- Extracts 1024-dimensional feature vector for each audio
- Collects corresponding labels
- Stacks into numpy arrays for model training

**Output:**
- `X_train`: (409, 1024) - 409 samples, 1024 features each
- `y_train`: (409,) - 409 continuous labels

---

## 7. Test Data Feature Extraction

```python
features_test = []
test_filenames = [] 
print("\nExtracting Test dataset features...")

for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    file_path = os.path.join(TEST_AUDIO_DIR, row['filename'] + ".wav")
    
    if not os.path.exists(file_path):
        continue
    
    feat = extract_features(file_path)
    
    if feat is not None:
        features_test.append(feat)
        test_filenames.append(row['filename'])

if features_test:
    X_test = np.stack(features_test)
    print(f"\n--- Test features extracted ---")
    print(f"Test features shape: {X_test.shape}")
```

**Key Differences:**
- No labels available (this is the prediction set)
- Saves original filenames for submission file
- 197 test samples processed

**Output:**
- `X_test`: (197, 1024) - 197 samples for prediction

---

## 8. Hyperparameter Optimization with Optuna

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import numpy as np

# Data Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Objective Function
def objective(trial):
    params = {
        'objective': 'regression_l2',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        
        'n_estimators': trial.suggest_int('n_estimators', 100, 700),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return score.mean()

# Run Optimization
study = optuna.create_study(direction='maximize')
print("Starting Optuna optimization...")
study.optimize(objective, n_trials=10, show_progress_bar=True)
print("Optimization finished.")

# Get Best Results
print("\nBest trial:")
trial = study.best_trial
print(f"  Value (Negative MSE): {trial.value}")
print("  Best Hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train Final Model
best_params = trial.params
best_model = lgb.LGBMRegressor(
    objective='regression_l2',
    metric='rmse',
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    **best_params
)
best_model.fit(X_train, y_train)

# Evaluate on Validation Set
val_preds = best_model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_preds, squared=False)
val_mae = mean_absolute_error(y_val, val_preds)

print(f"\n--- Final Model Evaluation ---")
print(f"Validation RMSE: {val_rmse}")
print(f"Validation MAE:  {val_mae}")
```

**Optuna Optimization Explained:**

### Search Space Definition:
- **n_estimators**: Number of boosting rounds (100-700)
- **learning_rate**: Step size shrinkage (0.01-0.3, logarithmic scale)
- **max_depth**: Maximum tree depth (3-10)
- **num_leaves**: Maximum leaves per tree (5-30, reduced for small dataset)
- **subsample**: Row sampling ratio (0.6-1.0)
- **colsample_bytree**: Feature sampling ratio (0.6-1.0)
- **reg_alpha**: L1 regularization (0.0-1.0)
- **reg_lambda**: L2 regularization (0.0-1.0)

### Optimization Strategy:
- Uses **Tree-structured Parzen Estimator (TPE)** algorithm
- Tries 10 different hyperparameter combinations
- Each trial uses 5-fold cross-validation
- Maximizes negative MSE (minimizes MSE)

### Cross-Validation:
- Splits data into 5 folds
- Trains on 4 folds, validates on 1
- Repeats 5 times with different validation fold
- Returns average performance

**Best Parameters Found:**
```
n_estimators: 382
learning_rate: 0.154
max_depth: 4
num_leaves: 5
subsample: 0.708
colsample_bytree: 0.828
reg_alpha: 0.108
reg_lambda: 0.153
```

**Performance:**
- Cross-validation score: -0.351 (negative MSE)
- Validation RMSE: 0.586
- Validation MAE: 0.465

---

## 9. Model Saving

```python
import joblib
from sklearn.metrics import mean_squared_error

# Evaluate on validation set
val_preds = best_model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_preds, squared=False)

# Print Best Parameters
print("Best Hyperparameters:", best_params)
print("Validation RMSE:", val_rmse)

# Save the Model
model_filename = "best_model_lightgbm.joblib"
joblib.dump(best_model, model_filename)
print(f"Model saved to: {model_filename}")
```

**Why joblib?**
- Efficient serialization for scikit-learn compatible models
- Preserves exact model state including hyperparameters
- Can be loaded with `joblib.load()` for future predictions

---

## 10. Test Set Predictions

```python
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# Extract features for TEST set
print("Extracting TEST dataset features...")
test_features = []

for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    filename_with_ext = f"{row['filename']}.wav"
    file_path = os.path.join(TEST_AUDIO_DIR, filename_with_ext)
    
    try:
        feat = extract_features(file_path)
        test_features.append(feat)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
X_test = np.stack(test_features)

# Make Predictions
print("Making final predictions...")
test_preds = best_model.predict(X_test)

# Create Submission DataFrame
df_test_preds = pd.DataFrame()
df_test_preds["filename"] = df_test['filename']
df_test_preds['label'] = test_preds

print("Predictions DataFrame:")
print(df_test_preds.head())

# Save to CSV
df_test_preds.to_csv('submission.csv', index=False)
print("Submission file saved!")
```

**Prediction Pipeline:**
1. Uses trained model to predict on 197 test samples
2. Creates submission DataFrame with filename and predicted label
3. Saves to CSV for competition submission

**Sample Predictions:**
```
filename      label
audio_141    2.585
audio_114    3.820
audio_17     3.206
audio_76     4.786
audio_156    3.082
```

---

## Key Insights and Best Practices

### 1. Transfer Learning Approach
- Uses pre-trained WavLM model (trained on 94k hours of speech)
- Freezes feature extractor, only trains final regressor
- Significantly reduces training time and data requirements

### 2. Feature Engineering
- Raw audio → 1024-dimensional embeddings
- Temporal pooling (averaging) creates fixed-size representation
- Captures semantic audio information

### 3. Model Selection
- LightGBM chosen over neural networks for small dataset (409 samples)
- Gradient boosting excels with limited data
- Faster training and inference than deep learning alternatives

### 4. Regularization Strategy
- L1 (Lasso) + L2 (Ridge) regularization
- Reduced `num_leaves` (5 instead of default 31)
- Subsampling features and rows
- Prevents overfitting on small dataset

### 5. Validation Strategy
- 70-30 train-validation split
- 5-fold cross-validation during hyperparameter search
- Ensures robust performance estimates

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Training Samples | 409 |
| Test Samples | 197 |
| Feature Dimensions | 1024 |
| Validation RMSE | 0.586 |
| Validation MAE | 0.465 |

---

## Potential Improvements

### 1. Data Augmentation
- Time stretching, pitch shifting
- Adding background noise
- Could increase effective training data

### 2. Ensemble Methods
- Average predictions from multiple models (WavLM, HuBERT, data2vec)
- Could improve robustness

### 3. Advanced Pooling
- Attention-based pooling instead of mean
- Max pooling or learnable pooling
- May capture more relevant temporal information

### 4. Feature Engineering
- Add traditional audio features (MFCCs, spectral features)
- Combine with deep learning embeddings
- Could provide complementary information

### 5. More Hyperparameter Tuning
- Increase Optuna trials (currently 10)
- Explore other model types (CatBoost, XGBoost ensemble)

---

## Complete Pipeline Architecture

```
┌─────────────────┐
│  Audio Files    │
│   (.wav)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Librosa Load   │
│  (16kHz)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  WavLM-Large    │
│  Feature        │
│  Extractor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1024-dim       │
│  Embeddings     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LightGBM       │
│  Regressor      │
│  (Optimized)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │
│  (Labels)       │
└─────────────────┘
```

---

## Conclusion

This pipeline demonstrates a **modern approach to audio classification** combining:
- State-of-the-art pre-trained models (WavLM)
- Efficient traditional ML (LightGBM)
- Automated hyperparameter optimization (Optuna)

The approach is particularly well-suited for **small audio datasets** where full end-to-end deep learning would likely overfit.

### Why This Approach Works:

✅ **Transfer Learning**: Leverages 94k hours of pre-training  
✅ **Efficiency**: Fast inference and training  
✅ **Robustness**: Cross-validation and regularization prevent overfitting  
✅ **Simplicity**: Clean, maintainable code  
✅ **Performance**: Competitive results with minimal data  

---

## Quick Reference Commands

```bash
# Install dependencies
pip install whisper protobuf==3.20.3
apt-get install -y libsndfile1

# Load model
processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
model = AutoModel.from_pretrained("microsoft/wavlm-large").to(DEVICE).eval()

# Extract features
features = extract_features(audio_path)

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Train model
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)

# Save submission
df_test_preds.to_csv('submission.csv', index=False)
```

---

**Total Training Time**: ~30 minutes on GPU (V100)  
**Inference Time**: ~2 seconds per audio file  
**Model Size**: 1.26GB (WavLM) + 500KB (LightGBM)
