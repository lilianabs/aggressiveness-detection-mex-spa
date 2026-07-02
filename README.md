# Aggressiveness Detection in Mexican Spanish

This repository implements a binary text classification pipeline to detect aggressive tweets in Mexican Spanish, based on the [MEX-A3T dataset](https://ceur-ws.org/Vol-2664/mexa3t_paper9.pdf).

**Task**: Binary classification on the MEX-A3T dataset — Category 0 (not aggressive) vs. Category 1 (aggressive).  
**Dataset**: ~7332 tweets (~71% non-aggressive, ~29% aggressive — class imbalanced).  
**Key metric**: Recall on the aggressive class (Category 1), since false negatives (missed aggressive tweets) are costly.

## Setup

### Requirements

- Python >=3.13
- Dependencies listed in `requirements.txt`

### Installation

1. Clone or enter the repository directory.

2. Install dependencies (choose one):

   **Using `uv` (recommended, faster):**
   ```bash
   uv sync
   ```

   **Using `pip`:**
   ```bash
   pip install -e .
   ```

3. Download the spaCy Spanish language model (required for stopword removal):
   ```bash
   python -m spacy download es_core_news_sm
   ```

4. **Optional: Set up Weights & Biases experiment tracking**

   Create a `.env` file in the project root with your Weights & Biases API key:
   ```
   WANDB_API_KEY=<your-wandb-api-key>
   ```

   (If you don't have a W&B account, skip this — the pipeline will run without logging.)

## Running the Pipeline

The project includes three main scripts that form the complete ML pipeline:

### 1. Preprocess Raw Data

```bash
python scripts/run_preprocess_data.py
```

- Loads raw data from `data/train_aggressiveness.csv`
- Applies all cleaning steps from `config.yaml` (emoji removal, special token removal, stopword removal, punctuation removal, lowercase, whitespace normalization)
- Outputs cleaned text to `data/train_aggressiveness_cleaned.csv`

**Output:** `data/train_aggressiveness_cleaned.csv`

### 2. Train and Evaluate Baseline Model (SVM)

```bash
python scripts/run_train_baseline_model.py
```

- Loads preprocessed data
- Splits into train/test (80/20 stratified split, `random_state=42`)
- Vectorizes with TF-IDF (n-grams: 1–3)
- Performs 5-fold stratified cross-validation on training set
- Trains SVM (linear kernel) on full training set
- Evaluates on test set
- Prints metrics: accuracy, precision, recall, F1
- **Saves test predictions to `data/test_predictions.csv`** (includes raw text, preprocessed text, true label, prediction)
- Logs results to Weights & Biases under the `aggressiveness-detection` project

**Outputs:**
- `data/test_predictions.csv` — predictions for error analysis
- Experiment logs in W&B (if `.env` is configured)

### 3. Compare Multiple Models

```bash
python scripts/run_models_experiment.py
```

- Uses the same preprocessed data and TF-IDF vectorization as the baseline
- Trains and evaluates three models for comparison:
  - **Logistic Regression** (max_iter=1000)
  - **Random Forest** (100 trees)
  - **Naive Bayes** (Multinomial)
- For each model:
  - Performs 5-fold stratified cross-validation
  - Trains on full training set
  - Evaluates on test set
  - Logs CV and test metrics to Weights & Biases
- Prints metrics for each model

**Output:** Experiment logs in W&B

## Project Structure

```
.
├── README.md                           # This file
├── CLAUDE.md                           # Technical documentation
├── config.yaml                         # Configuration (data paths, preprocessing steps, hyperparameters)
├── pyproject.toml                      # Project metadata and dependencies
│
├── src/
│   ├── preprocess_data.py              # Text cleaning functions
│   ├── train_model.py                  # Dataset splitting, vectorization, cross-validation
│   ├── evaluate.py                     # Metrics computation and prediction saving
│   └── utils.py                        # Config loading, path resolution, CSV I/O
│
├── scripts/
│   ├── run_preprocess_data.py          # Step 1: Clean raw data
│   ├── run_train_baseline_model.py     # Step 2: Train baseline SVM + save predictions
│   └── run_models_experiment.py        # Step 3: Benchmark multiple models
│
├── notebooks/
│   ├── Explotatory_data_analysis.ipynb # EDA: class distribution, text length, word clouds
│   ├── Baseline_model.ipynb            # Reference notebook (manual pipeline)
│   ├── Pipeline_for_experimenting.ipynb# Exploration with sklearn pipelines
│   └── Error_analysis.ipynb            # Inspect misclassified examples (from test_predictions.csv)
│
├── data/
│   ├── train_aggressiveness.csv        # Raw dataset (~730 KB)
│   ├── train_aggressiveness_cleaned.csv # Preprocessed dataset (~411 KB)
│   └── test_predictions.csv            # Test set predictions (generated after Step 2)
│
└── tests/
    └── test_preprocess_data.py         # Unit tests for cleaning functions
```

## Experiments

### 1. Exploratory Data Analysis

**File:** `notebooks/Explotatory_data_analysis.ipynb`

Initial exploration of the MEX-A3T dataset:
- Class distribution (~70% / ~30% split)
- Tweet length statistics
- Word clouds for each class

### 2. Baseline Model: SVM

**File:** `scripts/run_train_baseline_model.py`

Established a baseline classifier using Support Vector Machine (SVM):
- **Features**: TF-IDF vectorization (n-grams: 1–3, fit on training set only)
- **Model**: SVM with linear kernel (`C=1.0`, `random_state=42`)
- **Evaluation**:
  - 5-fold stratified cross-validation on training data
  - Test set evaluation (20% held out)
  - Metrics: accuracy, precision, recall (weighted + for class 1), F1-score
- **Predictions saved**: `data/test_predictions.csv` includes both raw and preprocessed text for error inspection

### 3. Error Analysis

**File:** `notebooks/Error_analysis.ipynb`

Detailed inspection of model mistakes:
- Loads `data/test_predictions.csv`
- Filters false positives (predicted aggressive, actually not) and false negatives (predicted not aggressive, actually is)
- Reads raw tweet text alongside model predictions for qualitative analysis
- Can generate confusion matrix heatmaps and classification reports

### 4. Model Comparison

**File:** `scripts/run_models_experiment.py`

Benchmark three additional algorithms against the SVM baseline using identical preprocessing and evaluation:

| Model | Details |
|-------|---------|
| **Logistic Regression** | max_iter=1000, L2 penalty, `random_state=42` |
| **Random Forest** | 100 trees, default hyperparameters, `random_state=42` |
| **Naive Bayes** | Multinomial (works with TF-IDF non-negative counts) |

**Evaluation protocol**: Same as baseline (5-fold CV + test set eval).

**Tracking**: All results logged to Weights & Biases (`aggressiveness-detection` project) for easy cross-run comparison.

## Configuration

**File:** `config.yaml`

Centralized configuration for reproducibility:

```yaml
data:
  local_raw_data_path: "data/train_aggressiveness.csv"
  local_preprocessed_data_path: "data/train_aggressiveness_cleaned.csv"
  local_predictions_path: "data/test_predictions.csv"

reproducibility:
  train_test_split:
    test_size: 0.2
    random_state: 42
    stratify: true

task:
  text_column: "Text"
  label_column: "Category"
  preprocessing_steps:
    - remove_emojis
    - remove_special_tokens
    - lowercase
    - remove_punctuation
    - remove_stopwords
    - remove_extra_whitespace
```

**Note:** Preprocessing order matters. `remove_special_tokens` must run *before* `remove_punctuation` to avoid partial token removal (e.g., `<URL>` → `<UR` if punctuation removal runs first).

## Key Insights

- **Class imbalance**: The dataset is skewed (~71% non-aggressive). Stratified splits ensure train/test preserve this distribution; recall on the minority class (aggressive) is prioritized.
- **Feature importance**: TF-IDF with n-grams (1–3) captures both unigrams and context (bigrams/trigrams). This outperformed unigrams-only in baseline tests.
- **Cross-validation**: Stratified k-fold (k=5) is used to reliably estimate model performance on imbalanced data and to detect overfitting.
- **Metric selection**: Accuracy alone is misleading on imbalanced data. Always inspect precision, recall, and F1, especially for the minority class.

## Next Steps

Possible improvements and future work:
- Try other feature representations (word embeddings, contextual models like transformers)
- Hyperparameter tuning (GridSearchCV or Optuna)
- Ensemble methods combining multiple models
- Data augmentation to address class imbalance
- Neural network approaches (deep learning)

## References

- [MEX-A3T Dataset Paper](https://ceur-ws.org/Vol-2664/mexa3t_paper9.pdf)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Weights & Biases](https://wandb.ai/)
