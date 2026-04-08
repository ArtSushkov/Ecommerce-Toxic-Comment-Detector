# 🛡️ Ecommerce Toxic Comment Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-orange)
![BERT](https://img.shields.io/badge/Model-Transformer-red)

> Automated content moderation system for Wikishop, leveraging traditional NLP pipelines and Transformer-based embeddings to detect toxic user comments and protect community integrity.

## 📖 Project Overview
**Wikishop** is launching a wiki-style editing platform where users can modify product descriptions and comment on each other's edits. To maintain a safe and constructive environment, the business requires an automated tool to flag toxic or offensive comments for manual moderation.

**Primary Goal:** Build a binary classification model to identify toxic comments with an **F1-score ≥ 0.75**.

## 🎯 Business Context & Challenges
- 📉 **Class Imbalance:** ~90% of comments are non-toxic, requiring careful handling to avoid biased predictions.
- 🌐 **Noisy User Input:** Comments contain typos, slang, repetitive text, and varying lengths.
- ⚖️ **Trade-off:** Balance precision (avoiding false moderation flags) and recall (catching truly toxic content).

## 📊 Dataset
| Column | Description |
|:-------|:------------|
| `text` | Raw user comment (unstructured text) |
| `toxic` | Target variable: `0` (non-toxic), `1` (toxic) |

**Volume:** ~159,292 labeled comments  
**Preprocessing Steps:**
- Text cleaning (regex, lowercasing, punctuation removal)
- Lemmatization via `spaCy` (English model)
- Removal of statistical anomalies (`word_count > 400`, `avg_word_length > 10`)
- Feature engineering: `word_count`, `avg_word_length` (evaluated but found non-informative)

## 🛠 Methodology & Pipeline
1. **Data Preparation & EDA**  
   - Handled class imbalance via `class_weight='balanced'` and threshold optimization
   - Analyzed text length distributions, word clouds, and lexical patterns
2. **Traditional NLP Baselines**  
   - `TF-IDF` vectorization (unigrams, stopword filtering)
   - Models: `LogisticRegression`, `XGBoost`
   - Hyperparameter tuning via `RandomizedSearchCV`
   - **Threshold tuning** on validation probabilities to maximize F1
3. **Deep Learning / Transformers**  
   - `unitary/toxic-bert` pretrained model
   - Extracted `[CLS]` embeddings for a stratified 1% subset (~1,000 samples) due to compute constraints
   - Fine-tuned `LogisticRegression` on top of BERT embeddings

## 📈 Results & Model Performance
| Model Approach | Validation F1 | Test F1 | Notes |
|:---------------|:-------------:|:-------:|:------|
| TF-IDF + LogisticRegression (default threshold) | 0.7501 | 0.7479 | Meets baseline requirement |
| TF-IDF + LogisticRegression (**tuned threshold**) | - | **~0.763** | +1.5% gain via threshold optimization |
| Toxic-BERT + LogisticRegression (1% data) | 0.8939 | **0.8333** | Superior performance with minimal data |

✅ **Final Recommendation:** `Toxic-BERT` pipeline for production due to higher robustness, contextual understanding, and significantly better F1-score. TF-IDF + LR serves as a lightweight, interpretable fallback for low-resource environments.

## 💡 Key Insights & Business Recommendations
🔍 **Lexical Patterns:** Toxic comments heavily concentrate around explicit slurs, aggressive pronouns, and repetitive phrasing.  
📋 **Actionable Recommendations:**
1. **Deploy BERT-based Classifier:** Integrate the transformer model into the moderation queue for highest accuracy.
2. **Implement Confidence Thresholding:** Route low-confidence predictions (`0.3 < prob < 0.7`) to human moderators to reduce false positives/negatives.
3. **Continuous Monitoring:** Track drift in comment toxicity distribution as user base grows; retrain quarterly or upon performance decay.

## 📁 Project Structure
