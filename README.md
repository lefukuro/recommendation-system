# Recommendation System Project

## 📋 Project Overview
A machine learning system for personalized content recommendations based on user engagement data. The project demonstrates full ML pipeline from data preprocessing to model deployment.

## 🎯 Business Problem
Increase user engagement by providing relevant content recommendations using historical interaction data.

## 🛠️ Technical Stack
- **Python** (pandas, numpy, scikit-learn)
- **CatBoost** for gradient boosting
- **NLTK and TF-IDF** for text processing
- **PCA** for dimensionality reduction
- **transformers** for text embeddings

## 📊 Key Features
- EDA, Data preprocessing and feature engineering
- Advanced text processing with TF-IDF and PCA
- Hyperparameter tuning with RandomizedSearchCV
- Recommendation quality evaluation using ROC-AUC metric
- Feature importance analysis

## 📈 Results
- **NDCG@5 Score by test data**: 0.607
- **Key drivers**: User engagement history, content topics, user data, text features, user-topic interactions features

## 🚀 Quick Start with Docker
```bash
git clone https://github.com/lefukuro/recommendation-system.git
cd recommendation-system
docker build -t recsys .
docker run recsys
```

## 📝 License
MIT License
