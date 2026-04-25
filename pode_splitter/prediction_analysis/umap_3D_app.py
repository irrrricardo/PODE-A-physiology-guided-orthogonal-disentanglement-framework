## To run, type the following in the terminal (from the repository root):
# streamlit run PODE/pode_splitter/prediction_analysis/umap_3D_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Page layout
st.set_page_config(layout="wide", page_title="3D Interactive Fundus Aging Analyzer")

# ================= Core functions =================

@st.cache_data
def load_data(file_input, file_type='path'):
    """
    Load and preprocess data.
    file_input: either a file path string, or an uploaded file object.
    file_type: 'path' or 'uploaded'.
    """
    df = None
    try:
        # 1. Read data depending on the input type
        if file_type == 'path':
            if file_input.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif file_input.endswith('.xlsx'):
                df = pd.read_excel(file_input)
            else:
                st.error("Unsupported file format, please use .csv or .xlsx")
                return None, None
        elif file_type == 'uploaded':
            # Streamlit uploaded-file object
            if file_input.name.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif file_input.name.endswith('.xlsx'):
                df = pd.read_excel(file_input)
        
        if df is None:
            return None, None

        # 2. Auto-detect numeric columns for dimensionality reduction
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude meaningless ID columns (adjust based on your data characteristics)
        exclude_cols = ['id', 'ID', 'PatientID', 'Unnamed: 0'] 
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        return df, feature_cols
        
    except FileNotFoundError:
        st.error(f"File not found: {file_input}. Please check the path.")
        return None, None
    except Exception as e:
        st.error(f"Error while reading file: {e}")
        return None, None

@st.cache_data
def compute_umap(df, feature_cols, n_neighbors=15, min_dist=0.1):
    """Compute the UMAP 3D coordinates."""
    # Simple missing-value imputation
    data = df[feature_cols].fillna(df[feature_cols].mean())
    scaled_data = StandardScaler().fit_transform(data)
    
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=42,
        n_jobs=1
    )
    embedding = reducer.fit_transform(scaled_data)
    
    df_umap = df.copy()
    df_umap['UMAP_X'] = embedding[:, 0]
    df_umap['UMAP_Y'] = embedding[:, 1]
    df_umap['UMAP_Z'] = embedding[:, 2]
    return df_umap

# ================= UI logic =================

st.title("🧬 Multi-dimensional Physiological Aging Interactive Analysis System")

# --- Sidebar: data loading and configuration ---
with st.sidebar:
    st.header("📂 1. Data loading")
    
    # Option: choose between uploading a file or entering a path
    data_source_mode = st.radio(
        "Choose the data source:",
        ("Enter local file path", "Upload file directly"),
        index=0
    )

    df_raw = None
    feature_cols = None

    if data_source_mode == "Enter local file path":
        # Use your previous filename as the default for convenience
        default_path = "temp.xlsx - Sheet1.csv"
        file_path = st.text_input("Enter the absolute or relative file path:", value=default_path)
        
        if file_path and st.button("Load file from path"):
            if os.path.exists(file_path):
                df_raw, feature_cols = load_data(file_path, file_type='path')
                if df_raw is not None:
                    st.success(f"Successfully loaded: {len(df_raw)} rows of data")
            else:
                st.error("Path does not exist, please check.")
                
    else: # Upload file directly
        uploaded_file = st.file_uploader("Drag & drop a file here (CSV/Excel supported)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            df_raw, feature_cols = load_data(uploaded_file, file_type='uploaded')
            if df_raw is not None:
                st.success(f"Successfully uploaded: {len(df_raw)} rows of data")

    st.divider()

# ================= Main logic: only show subsequent content if data was loaded successfully =================

if df_raw is not None and feature_cols is not None:
    
    with st.sidebar:
        st.header("⚙️ 2. Parameter configuration")
        
        # --- Clustering / dimensionality-reduction settings ---
        st.subheader("3D space construction (clustering)")
        n_neighbors = st.slider("Number of clustering neighbors", 2, 50, 15, help="Smaller values focus on local structure, larger values focus on global structure")
        min_dist = st.slider("Minimum distance between points", 0.0, 1.0, 0.1)
        
        with st.spinner("Building 3D space..."):
            df_3d = compute_umap(df_raw, feature_cols, n_neighbors, min_dist)
        
        st.divider()

        # --- Coloring and filtering settings ---
        st.subheader("Dimension perspective (coloring & filtering)")
        
        # Smart preselection: prefer Delta-related columns, otherwise the first column
        default_idx = 0
        if 'Age_Delta_hemo' in feature_cols:
            default_idx = feature_cols.index('Age_Delta_hemo')
            
        color_col = st.selectbox("Select coloring dimension (Color By)", options=feature_cols, index=default_idx)
        
        # Dynamic range slider
        min_val = float(df_3d[color_col].min())
        max_val = float(df_3d[color_col].max())
        if np.isnan(min_val): min_val = 0.0
        if np.isnan(max_val): max_val = 1.0
            
        range_val = st.slider(
            f"Limit display range of {color_col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        
        # K-Means option
        use_kmeans = st.checkbox("Enable K-Means automatic-clustering coloring", value=False)
        if use_kmeans:
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_data = df_3d[feature_cols].fillna(0)
            df_3d['Cluster'] = kmeans.fit_predict(cluster_data).astype(str)
            color_target = 'Cluster'
        else:
            color_target = color_col
        
        st.divider()
        st.subheader("🎨 Visual style adjustment")
        point_size = st.slider("Point size", min_value=1, max_value=15, value=5, help="Adjust the scatter point size")
        point_opacity = st.slider("Point opacity", min_value=0.1, max_value=1.0, value=0.8, help="Adjust the scatter point opacity")

    # --- Main area plotting ---
    mask = (df_3d[color_col] >= range_val[0]) & (df_3d[color_col] <= range_val[1])
    df_filtered = df_3d[mask]
    df_hidden = df_3d[~mask]
    
    # Dynamic title
    chart_title = f"3D Physiological Feature Space - Coloring: {color_target}"
    if not use_kmeans:
        chart_title += f" (Range: {range_val[0]:.2f} ~ {range_val[1]:.2f})"

    fig = px.scatter_3d(
        df_filtered,
        x='UMAP_X', y='UMAP_Y', z='UMAP_Z',
        color=color_target,
        color_continuous_scale='Spectral_r' if not use_kmeans else None,
        hover_data=['id'] + feature_cols[:5], # show the first 5 features for hovering
        title=chart_title
    )
    
    # Use the slider values to update point size and opacity
    fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))
    
    # Add gray background points
    if not df_hidden.empty:
        fig.add_scatter3d(
            x=df_hidden['UMAP_X'], y=df_hidden['UMAP_Y'], z=df_hidden['UMAP_Z'],
            mode='markers',
            marker=dict(size=3, color='lightgray', opacity=0.15), # a bit lighter to avoid grabbing visual attention
            name='Filtered Out', hoverinfo='skip'
        )

    fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)
    
    # Bottom data preview
    st.markdown("### 📊 Filtered data preview")
    st.dataframe(df_filtered)

else:
    # Initial guide page
    st.info("👈 Please load your data file (CSV or Excel supported) from the left sidebar to start the analysis.")
