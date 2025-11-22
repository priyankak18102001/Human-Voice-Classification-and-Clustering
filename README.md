# Human-Voice-Classification-and-Clustering
Streamlit app for classifying and clustering human voice using ML.
# 🎤 Human Voice Classification & Clustering (Streamlit App)

This project predicts the **gender of a speaker** and **clusters similar voice samples** using machine learning on pre-extracted **numerical features** (e.g., pitch, spectral features, MFCCs).

The app is built with **Streamlit** and has a simple navigation layout:

1. **Introduction** – brief overview of the project  
2. **EDA Summary** – explains the exploratory data analysis performed  
3. **Prediction & Clustering** – manual input of features to get:
   - **Gender prediction** (Male/Female) using SVM  
   - **Cluster assignment** using KMeans  

> 🔹 Note: This app uses *numerical features only* (like `mean_pitch`, spectral features, MFCC means, etc.).  
> It does **not** take raw audio as input in the UI.

---

## 🧠 Problem Statement

Given a set of features extracted from speech (such as pitch, spectral centroid, MFCCs), we want to:

- **Classify** a voice as **Male** or **Female**
- **Cluster** voices into groups based on similarity, to analyze speaker patterns

This can be useful in:
- Speaker profiling  
- Voice analytics  
- Pre-processing for downstream speech tasks  

---
Classification – Support Vector Machine (SVM)

Input: ~14–20 scaled numerical features (e.g., mean_pitch, spectral features, MFCC stats)
Output: 0 = Female, 1 = Male
Trained on selected features after feature selection & scaling.
Evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion matrix
****
Clustering – KMeans

Input: same feature space as classifier (scaled features)
Output: cluster labels (e.g., Cluster 0, 1, 2, …)
Evaluated using:
Silhouette score
Cluster size distribution

## 🏗️ Project Structure

```text
Human_voice_model/
├─ app.py                          # Streamlit application (3-page UI)
├─ Human Voice Clustering and Classification.pdf   # Project report / documentation (optional)
├─ models/
│   ├─ best_model_svm.joblib       # Trained SVM classifier (gender prediction)
│   ├─ kmeans_best.joblib          # Trained KMeans clustering model
│   ├─ scaler_20_features.joblib   # StandardScaler fit on training data
│   └─ selected_features_20.pkl    # List of selected feature names (14–20)
├─ model_comparison_summary.csv    # Evaluation summary of models (optional)
├─ requirements.txt                # Python dependencies
├─ README.md                       # Project description (this file)
└─ .gitignore                      # Files/folders ignored by Git (env, cache, etc.)
