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

The project includes two main scripts that form the complete ML pipeline:

### 1. Preprocess Raw Data

```bash
uv run scripts/run_preprocess_data.py
```

- Loads raw data from `data/raw/train_aggressiveness.csv`
- Applies all cleaning steps from `config.yaml` (emoji removal, special token removal, lowercase, punctuation removal, stopword removal, whitespace normalization)
- Outputs cleaned text to `data/preprocessed/train_aggressiveness_cleaned.csv`

**Output:** `data/preprocessed/train_aggressiveness_cleaned.csv`

### 2. Train and Evaluate Models

**Option A: Train baseline SVM model**

```bash
uv run scripts/run_train_model.py
```

- Loads preprocessed data
- Splits into train/test (80/20 stratified split, `random_state=42`)
- Vectorizes with TF-IDF (n-grams: 1–4)
- Performs 5-fold stratified cross-validation on training set
- Trains SVM (linear kernel, balanced class weights) on full training set
- Evaluates on test set
- Prints metrics: accuracy, precision, recall, F1, confusion matrix, classification report
- **Saves test predictions to `data/predictions/test_predictions-SVM_ngram_1-4_class_weight_balanced_kernel_linear.csv`** (includes raw text, preprocessed text, true label, prediction)
- Logs results to Weights & Biases under the `aggressiveness-detection` project

**Outputs:**
- `data/predictions/test_predictions-{model_name}.csv` — predictions for error analysis
- Experiment logs in W&B (if `.env` is configured)

**Option B: Compare multiple models**

```bash
uv run scripts/run_train_models.py
```

- Uses the same preprocessed data and TF-IDF vectorization as the baseline
- Trains and evaluates three models for comparison:
  - **Logistic Regression** (max_iter=1000, balanced class weights)
  - **Random Forest** (100 trees, balanced class weights)
  - **Naive Bayes** (Multinomial)
- For each model:
  - Performs 5-fold stratified cross-validation on training set
  - Trains on full training set
  - Evaluates on test set
  - Saves predictions to `data/predictions/test_predictions-{model_name}.csv`
  - Logs CV and test metrics to Weights & Biases
- Prints metrics for each model

**Output:** 
- Experiment logs in W&B
- Predictions saved for each model for error analysis

**Option C: Combined word + character-level TF-IDF experiment**

```bash
uv run scripts/run_combine_character_level_n_grams.py
```

- Combines word-level and character-level TF-IDF features using `sklearn.pipeline.FeatureUnion`
- Word-level: TF-IDF with n-grams 1–4, analyzer='word'
- Character-level: TF-IDF with n-grams 2–5, analyzer='char_wb' (character n-grams within word boundaries)
- Trains SVM (linear kernel, balanced class weights) on concatenated feature space
- Performs 5-fold stratified cross-validation
- Evaluates on test set
- Saves predictions and logs metrics to Weights & Biases

**Output:** 
- `data/predictions/test_predictions-SVM_combined_word_1-4_char_wb_2-5_class_weight_balanced_kernel_linear.csv`
- Experiment logs in W&B with combined feature metrics

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
│   ├── train_model.py                  # Dataset splitting, vectorization, cross-validation, training pipeline
│   ├── evaluate_model.py               # Metrics computation and prediction saving
│   └── utils.py                        # Config loading, path resolution, CSV I/O, W&B logging
│
├── scripts/
│   ├── run_preprocess_data.py                        # Step 1: Clean raw data
│   ├── run_train_model.py                            # Step 2: Train baseline SVM + save predictions
│   ├── run_train_models.py                           # Step 2 (alt): Benchmark multiple models (Logistic Regression, Random Forest, Naive Bayes)
│   └── run_combine_character_level_n_grams.py       # Step 2 (alt): SVM with combined word + char TF-IDF features
│
├── notebooks/
│   ├── Explotatory_data_analysis.ipynb # EDA: class distribution, text length, word clouds
│   ├── Baseline_model.ipynb            # Reference notebook (manual pipeline)
│   ├── Pipeline_for_experimenting.ipynb# Exploration with sklearn pipelines
│   └── Error_analysis.ipynb            # Inspect misclassified examples (from test_predictions CSV)
│
├── data/
│   ├── raw/
│   │   └── train_aggressiveness.csv                    # Raw dataset (~730 KB)
│   ├── preprocessed/
│   │   └── train_aggressiveness_cleaned.csv           # Preprocessed dataset (~411 KB)
│   └── predictions/
│       └── test_predictions-{model_name}.csv          # Test set predictions (generated after Step 2)
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

**File:** `scripts/run_train_model.py`

Established a baseline classifier using Support Vector Machine (SVM):
- **Features**: TF-IDF vectorization (n-grams: 1–4, fit on training set only)
- **Model**: SVM with linear kernel, balanced class weights (`class_weight='balanced'`, `random_state=42`)
- **Evaluation**:
  - 5-fold stratified cross-validation on training data (preserves class distribution)
  - Test set evaluation (20% held out)
  - Metrics: accuracy, precision, recall, F1-score, confusion matrix, classification report
- **Predictions saved**: `data/predictions/test_predictions-{model_name}.csv` includes both raw and preprocessed text for error inspection
- **Weights & Biases logging**: Cross-validation metrics logged for tracking experiments

### 3. Error Analysis

**File:** `notebooks/Error_analysis.ipynb`

Detailed inspection of model mistakes:
- Loads `data/test_predictions.csv`
- Filters false positives (predicted aggressive, actually not) and false negatives (predicted not aggressive, actually is)
- Reads raw tweet text alongside model predictions for qualitative analysis
- Can generate confusion matrix heatmaps and classification reports

### 4. Model Comparison

**File:** `scripts/run_train_models.py`

Benchmark three additional algorithms alongside the SVM baseline using identical preprocessing and evaluation:

| Model | Details |
|-------|---------|
| **Logistic Regression** | max_iter=1000, L2 penalty, balanced class weights, `random_state=42` |
| **Random Forest** | 100 trees, balanced class weights, `random_state=42` |
| **Naive Bayes** | Multinomial (works with TF-IDF non-negative counts) |

**Evaluation protocol**: Same as baseline (5-fold stratified CV + test set eval).

**Tracking**: All results (CV metrics, predictions) logged to Weights & Biases (`aggressiveness-detection` project) for easy cross-run comparison.

### 5. Combined Feature Engineering

**File:** `scripts/run_combine_character_level_n_grams.py`

Experiment combining word-level and character-level TF-IDF features via `FeatureUnion`:
- **Word features**: TF-IDF with n-grams 1–4, analyzer='word'
- **Character features**: TF-IDF with n-grams 2–5, analyzer='char_wb' (word-boundary-aware character n-grams)
- **Model**: SVM with linear kernel and balanced class weights
- **Rationale**: Character-level n-grams capture spelling patterns and subword information, complementing word-level features for better aggressiveness detection

**Evaluation protocol**: Same as baseline (5-fold stratified CV + test set eval).

**Tracking**: Results logged to Weights & Biases for comparison with single-feature baselines.

## Configuration

**File:** `config.yaml`

Centralized configuration for reproducibility:

```yaml
project:
  name: "aggressiveness-detection"

data:
  local_raw_data_path: "data/raw/train_aggressiveness.csv"
  local_preprocessed_data_path: "data/preprocessed/train_aggressiveness_cleaned.csv"
  local_predictions_file: "data/predictions/test_predictions-{model_name}.csv"

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

## Experiment Results Summary

### Cross-Validation Performance (5-fold Stratified)

All experiments use 5-fold stratified cross-validation on the training set. Below is a summary of key runs:

| Model | Features | CV Accuracy | CV Recall | CV F1 | Notes |
|-------|----------|-------------|-----------|-------|-------|
| **SVM (combined)** | word(1-4) + char_wb(2-5) | 0.829 ± 0.010 | 0.613 ± 0.028 | 0.673 ± 0.022 | **Best accuracy** — combines word + char features |
| SVM (char_wb 1-4) | char_wb n-grams 1–4 | 0.807 ± 0.010 | 0.723 ± 0.015 | 0.683 ± 0.015 | Character-level only; high recall on aggr. class |
| SVM (word 1-2) | word n-grams 1–2 | 0.825 ± 0.008 | 0.581 ± 0.030 | 0.656 ± 0.019 | Word-level baseline |
| SVM (char 1-4) | char n-grams 1–4 | 0.812 ± 0.010 | 0.731 ± 0.024 | 0.691 ± 0.017 | Full-text character n-grams |
| Logistic Regression (1-3) | word n-grams 1–3 | 0.814 ± 0.008 | 0.600 ± 0.019 | 0.650 ± 0.011 | — |
| Random Forest (1-3) | word n-grams 1–3 | 0.822 ± 0.012 | 0.562 ± 0.047 | 0.645 ± 0.026 | — |

**Key observations:**
- **Combined features (word + char)** achieve the highest cross-validation accuracy (0.829) while maintaining reasonable recall on the aggressive class.
- **Character-level analyzers** (especially `char_wb`) tend to have higher recall on the minority (aggressive) class, important for this imbalanced dataset.
- **Word-level features** generally sacrifice recall for higher precision.

### Tracking & Reproducibility

All experiments are logged to [Weights & Biases](https://wandb.ai/) under the `aggressiveness-detection` project. Each run includes:
- Cross-validation metrics (accuracy, precision, recall, F1 with mean ± std)
- Preprocessing steps applied
- Feature extraction configuration
- Model hyperparameters

## Key Insights

- **Class imbalance**: The dataset is skewed (~71% non-aggressive). Stratified splits ensure train/test preserve this distribution; recall on the minority class (aggressive) is prioritized.
- **Feature engineering**: Combining word-level and character-level TF-IDF features improves overall accuracy. Character n-grams capture spelling patterns and subword units useful for detecting aggressive language.
- **Cross-validation**: Stratified k-fold (k=5) is used to reliably estimate model performance on imbalanced data and to detect overfitting.
- **Metric selection**: Accuracy alone is misleading on imbalanced data. Always inspect precision, recall, and F1, especially for the minority class.
- **Analyzer choice**: `char_wb` (character n-grams within word boundaries) balances capturing subword patterns while avoiding excessive sparsity compared to full-text character n-grams.

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
