import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# Header
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendations")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('online_retail.csv')
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose:", ["Dashboard", "Segmentation", "Recommendations"])

if page == "Dashboard":
    st.header("📊 Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{df['CustomerID'].nunique():,}")
    with col2:
        st.metric("Products", f"{df['StockCode'].nunique():,}")
    with col3:
        st.metric("Sales", f"${df['TotalAmount'].sum():,.0f}")
    with col4:
        st.metric("Countries", f"{df['Country'].nunique()}")
    
    # Sales by country
    country_sales = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=country_sales.values, y=country_sales.index, orientation='h', 
                 title="Top Countries by Sales")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Segmentation":
    st.header("🎯 Customer Segmentation")
    
    # Calculate RFM
    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Clustering with outlier handling
    scaler = StandardScaler()
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Handle outliers using IQR method
    Q1 = rfm_features.quantile(0.25)
    Q3 = rfm_features.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((rfm_features < (Q1 - 1.5 * IQR)) | (rfm_features > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Scale the data (only non-outliers)
    rfm_scaled = scaler.fit_transform(rfm_features[outlier_mask])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    
    # Add clusters to RFM data (only for non-outliers)
    rfm_clean = rfm[outlier_mask].copy()
    rfm_clean['Cluster'] = cluster_labels
    
    # Segment names
    segments = {0: 'High-Value', 1: 'Regular', 2: 'Occasional', 3: 'At-Risk'}
    rfm_clean['Segment'] = rfm_clean['Cluster'].map(segments)
    
    # Segment analysis
    st.subheader("Segment Analysis")
    segment_stats = rfm_clean.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)
    st.dataframe(segment_stats)
    
    # Segment distribution
    segment_counts = rfm_clean['Segment'].value_counts()
    fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                 title="Customer Segments Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictor
    st.subheader("🔮 Segment Predictor")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", value=30)
    with col2:
        frequency = st.number_input("Frequency", value=5)
    with col3:
        monetary = st.number_input("Monetary ($)", value=100.0)
    
    if st.button("Predict"):
        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_data)[0]
        predicted = segments[cluster]
        st.success(f"Predicted Segment: {predicted}")

elif page == "Recommendations":
    st.header("🤖 Product Recommendations")
    
    # Build recommendation system
    customer_product = df.pivot_table(
        index='CustomerID', columns='StockCode', 
        values='Quantity', aggfunc='sum', fill_value=0
    )
    
    similarity_matrix = cosine_similarity(customer_product.T)
    product_index = customer_product.columns
    product_names = df.set_index('StockCode')['Description'].drop_duplicates()
    
    # Product selector
    unique_products = df[['StockCode', 'Description']].drop_duplicates()
    product_options = {f"{row['StockCode']} - {row['Description'][:50]}": row['StockCode'] 
                      for _, row in unique_products.head(50).iterrows()}
    
    selected = st.selectbox("Select product:", list(product_options.keys()))
    
    if selected and st.button("Get Recommendations"):
        product_code = product_options[selected]
        if product_code in product_index:
            idx = list(product_index).index(product_code)
            scores = list(enumerate(similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
            
            st.success("Top 5 Similar Products:")
            for i, (idx, score) in enumerate(scores, 1):
                code = product_index[idx]
                name = product_names.get(code, f"Product {code}")
                st.write(f"{i}. {code} - {name} (Score: {score:.3f})")
    
    # Top products
    st.subheader("📈 Top Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=top_products.values, y=[desc[:30] for desc in top_products.index], 
                 orientation='h', title="Top Products by Quantity")
    st.plotly_chart(fig, use_container_width=True) 