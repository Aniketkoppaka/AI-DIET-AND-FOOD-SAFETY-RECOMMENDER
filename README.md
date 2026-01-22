# AI‑Driven Diet & Food Safety Recommender

An end‑to‑end prototype that blends food safety risk modeling with nutrition profiling and user preferences to recommend safer, healthier foods. It trains an ensemble model on EU RASFF alerts, merges with nutrition data, and personalizes recommendations based on dietary goals, allergies, and preferences.

---

## 📁 Project Structure

```
AI-DIET-AND-FOOD-SAFETY-RECOMMENDER/
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── notebooks/                 # Jupyter notebooks
│   ├── Diet+FoodSafetyModel.ipynb   # Main integrated system (recommended)
│   ├── DietRecommender.ipynb        # Diet recommendation only
│   └── FoodSafetyModel.ipynb        # Food safety model only
│
├── src/                       # Python scripts
│   ├── sota_pipeline.py             # SOTA model comparison (9 models)
│   ├── enhanced_pipeline.py         # Enhanced pipeline (6 models)
│   ├── comparative_analysis.py      # Analytics utilities
│   ├── run_analysis.py              # Run analysis demo
│   └── demo_example.py              # Demo showing input/output
│
├── DATASETS/                  # Data files
│   ├── DIET/                        # Diet-related data
│   ├── FOOD-CHOICES/                # User profiles
│   ├── FOOD-SAFETY/                 # RASFF contamination data
│   ├── MERGED/                      # Pre-scored master dataset
│   └── NUTRITION/                   # Nutritional information
│
└── outputs/                   # Generated results (CSVs, PNGs)
```

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

Open any notebook in VS Code, Jupyter, or Google Colab:

| Notebook | Description |
|----------|-------------|
| `notebooks/Diet+FoodSafetyModel.ipynb` | **Recommended** - Complete end-to-end system |
| `notebooks/FoodSafetyModel.ipynb` | Food safety risk prediction only |
| `notebooks/DietRecommender.ipynb` | Diet recommendations only |

> **Note**: Run notebooks from the project root directory for data paths to work correctly.

### Running Python Scripts

```bash
# Demo showing input/output format
python src/demo_example.py

# SOTA model comparison (9 models including DeBERTa, RoBERTa)
python src/sota_pipeline.py

# Enhanced pipeline (6 models)
python src/enhanced_pipeline.py

# Analysis with visualizations
python -m src.run_analysis
```

---

## 📊 What's Inside

### Notebooks

1. **`Diet+FoodSafetyModel.ipynb`** — Integrated End-to-End System
   - Trains safety model, scores products, generates personalized recommendations
   - Creates unified `recommendation_score` = 60% safety + 40% health

2. **`FoodSafetyModel.ipynb`** — Food Safety Risk Prediction
   - Ensemble model (XGBoost + LightGBM + Logistic Regression) on RASFF alerts
   - ~74% accuracy on food contamination risk classification

3. **`DietRecommender.ipynb`** — Personalized Diet Recommendations
   - Considers dietary restrictions, allergies, and health goals
   - Generates meal plans based on user profiles

### Datasets

| Dataset | Description | Rows |
|---------|-------------|------|
| `FOOD-SAFETY/rasff_cleaned_ready.csv` | EU RASFF contamination alerts | ~30k |
| `NUTRITION/nutrition_cleaned_ready.csv` | Nutritional info per 100g | ~400k |
| `FOOD-CHOICES/synthetic_users_cleaned_ready.csv` | User profiles | 500 |
| `MERGED/master_food_safety_recommender.csv` | Pre-scored master data | ~99k |

### SOTA Model Comparison

The `sota_pipeline.py` compares 9 models:

| Model | Type | Accuracy |
|-------|------|----------|
| DeBERTa-v3-large | Transformer | **74.5%** |
| RoBERTa-large | Transformer | 72.0% |
| DeBERTa-v3-base | Transformer | 71.7% |
| CatBoost | ML | 68.2% |
| XGBoost | ML | 67.0% |

---

## 🎯 Why It Matters

Most diet recommenders optimize for calories or macros only. Foodborne risks (pathogens, heavy metals, toxins) aren't considered. This project fuses a learned safety model with nutrition data to deliver recommendations that are both **safe** and **healthy**.

---

## 📜 License

Educational and research purposes. Check individual dataset licenses before commercial use.

---

## 🤝 Contributing

Contributions welcome! Open issues or submit pull requests.
