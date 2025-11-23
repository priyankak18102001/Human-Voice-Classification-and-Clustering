import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from pathlib import Path
import librosa, tempfile

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sns.set()
try:
    import umap
    UAMP = True
except Exception:
    UAMP = False

# ----------------- CONFIG: update filenames if needed -----------------
MODEL_PATH = "best_model_svm.joblib"
KMEANS_PATH = "kmeans_best.joblib"
SCALER_PATH = "scaler_20_features.joblib"
FEATURES_PATH = "selected_features_20.pkl"
#PROJECT_PDF = "C:/Users/PRIYANKA/OneDrive/Desktop/Human_voice_model/Human_voice_classification_and_clustring (1).ipynb" # project doc path
PROJECT_PDF =  "Human_voice_classification_and_clustring (1).ipynb"

# ---------------------------------------------------------------------

st.set_page_config(page_title="Voice Gender App",layout = "wide")
st.title("Human Voice Classification and Clustring")

# load artifacts (graceful)
@st.cache_resource

def load_artifact():
    artifacts = {}
    # model
    artifacts['model'] = None
    artifacts['kmeans'] = None
    artifacts['scaler'] = None
    artifacts['features'] =None

    try:
        artifacts['model'] = joblib.load(MODEL_PATH)
    except Exception as e:
        artifacts['model_error'] =  str(e)
    try:
        artifacts['kmeans'] = joblib.load(KMEANS_PATH)
    except Exception as e:
        artifacts['kmeans_error'] = str(e) 
    try:
        artifacts['scaler'] = joblib.load(SCALER_PATH)
    except Exception as e:
        artifacts['scaler_error'] = str(e)
    try:
        artifacts['features'] = joblib.load(FEATURES_PATH)
    except Exception as e:
        artifacts['features_error'] = str(e)

    return artifacts

art = load_artifact()

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Introduction", "EDA", "Prediction & Clustering"])



# ----------------- PAGE: INTRODUCTION -----------------
if page == "Introduction":
    st.header("Project Introduction")
    st.markdown("""  **Goal:** Build models to classify gender from voice samples and to cluster similar voice samples.
    - Classification model (SVM) predicts male/female.
    - Clustering model (KMeans) groups similar voices.
                """)
    if Path(PROJECT_PDF).exists():
        st.markdown(f"project document : {PROJECT_PDF}")
        if st.button("open project pdf(local)"):
            st.write("pdf path",PROJECT_PDF)
    else:
        st.info("project pdf is not found at expeceted path")  

# ----------------- PAGE: EDA -----------------
elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Upload your dataset (CSV) containing the feature columns & optional `label` column to run EDA.")
    path = "vocal_gender_features_new.csv"
    if path is not  None:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        st.write("Dataset preview")
        st.dataframe(df.head())

    st.write("#### 1. Distribution Plot (Histogram) – Pitch Mean (fundamental frequency)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df,x='mean_pitch',hue='label',kde=True,ax=ax)
    st.pyplot(fig)
    st.write("Look for separation / overlap — strong separability ⇒ pitch is highly predictive.")

    st.write("#### 2. Boxplot Pitch Mean by Gender")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df,x='label',y='mean_pitch',ax=ax)
    plt.title("Pitch Mean by Gender")
    plt.show()
    st.pyplot(fig)
    st.write("Compare medians & spread — wider female IQR and higher median expected.")

    st.write("#### 3. Violin Plot MFCC_1_mean vs Gender")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df,x='label',y='mfcc_1_mean',inner='quartile',ax=ax)
    plt.title("MFCC_1 Mean Distribution by Gender")
    plt.show()
    st.pyplot(fig)
    st.write("MFCC_1 captures spectral slope,Helps detect tonal differences in gender.")
    
    st.write("#### 4. Density Plot (KDE)  Spectral Centroid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df,x='mean_spectral_centroid',hue='label',fill=True,ax=ax)
    plt.title("Spectral Centroid Density by Gender")
    plt.show()
    st.pyplot(fig)
    st.write("Shows how brightness differs between genders.")

    st.write("#### 5. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0,ax=ax)
    plt.title("Correlation Heatmap")
    plt.show()
    st.pyplot(fig)
    st.write("Which features differ across gender.")

    st.write("####  6.Pairplot (scatter matrix)")
    fig, ax = plt.subplots(figsize=(6, 4))
    label_col='label'
    pair_features  = ['mean_spectral_centroid','rms_energy','std_spectral_centroid','mean_pitch','mfcc_1_mean','mfcc_1_std']
    pair_df = df[pair_features + [label_col]]
    pairplot = sns.pairplot(pair_df, hue=label_col, diag_kind="kde")
    st.pyplot(pairplot.fig)
    st.write("Shows natural separation line.")

    st.write("####7. PCA 2D Scatter Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    features = df.drop(columns='label')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['label'],ax=ax)
    plt.title("PCA 2D Scatter (Gender)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    st.pyplot(fig)
    st.write("Plot first 2 principal components, color by class.")

    st.write("####8.  UMAP or t-SNE Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df['label'],ax=ax)
    plt.title("t-SNE 2D Embedding")
    plt.show()
    st.pyplot(fig)
    st.write("Better than PCA for non-linear separation.")

    st.write("####9.  Cluster Heatmap (Feature Sample)")
    fig, ax = plt.subplots(figsize=(6, 4))
    cluster_map = sns.clustermap(df[features.columns].corr(), cmap="coolwarm")
    st.pyplot(cluster_map.fig)
    st.write("Shows patterns like similar MFCC behavior per speake Excellent for multi-class EDA")

# ----------------- PAGE: PREDICTION & CLUSTERING -----------------
elif page == "Prediction & Clustering":
    st.header("Classification Prediction & Clustering")

    if art.get("features") is None:
        st.error("Feature list not found. Make sure models/selected_features_20.pkl exists.")
    elif art.get("model") is None:
        st.error("SVM model not found. Make sure models/best_model_svm.joblib exists.")
    elif art.get("kmeans") is None:
        st.error("KMeans model not found. Make sure models/kmeans_best.joblib exists.")
    elif art.get("scaler") is None:
        st.error("Scaler not found. Make sure models/scaler_20_features.joblib exists.")
    else:
        model = art["model"]
        kmeans = art["kmeans"]
        scaler = art["scaler"]
        feature_names = art["features"]

        st.markdown("""
        Enter the values for each feature below.
        These features should be the **same 14/20 numeric features used when training the model**.
        """)    
           
        # If you only want to use first 14 of them, uncomment this line:
        # feature_names = feature_names[:14]

        # Use a form so that prediction happens only when we click the button
        with st.form("manual input form"):
            inputs =[]
            # layout: 2 columns of inputs
            cols = st.columns(2)
            for i,feat in enumerate(feature_names):
                col = cols[i%2]
                with col:
                    val = col.number_input(f"{feat}", value=0.0, format="%.4f")
                    inputs.append(val)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # create a single-sample array
            x_manual = np.array(inputs,dtype=float).reshape(1,-1)

            X_scaled = scaler.transform(x_manual)  
              # ---- Classification (SVM) ----
            try:
                y_pred = model.predict(X_scaled)
                if hasattr(model,"predict_proba"):
                   y_prob = model.predict_proba(X_scaled)[:,1]  
                else:
                    y_prob = None

                label_map = {0: "Female", 1: "Male"}
                gender = label_map.get(int(y_pred[0]),y_pred[0])

                st.subheader("Classification Result")
                st.success(f"Predicted Gender: **{gender}**")
                if y_prob is not None:
                    st.write(f"Confidence (probability of Male): {y_prob[0]:.3f}")

            except Exception as e:
                st.error(f"Error during classification prediction: {e}")
                gender = None        
                                 

            # ---- Clustering (KMeans) ----
            try:
                cluster_id = int(kmeans.predict(X_scaled)[0])
                st.subheader("Clustering Result (KMeans)")
                st.write(f"Assigned Cluster: **{cluster_id}**")

                # Show cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                cent_df = pd.Series(centroid, index=feature_names)
                st.write("Cluster centroid (in scaled feature space):")
                st.dataframe(cent_df.to_frame("centroid").T)
            except Exception as e:
                st.error(f"Error during clustering: {e}")










              




