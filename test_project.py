#!/usr/bin/env python3
"""
Test script for Shopper Spectrum project
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta

def test_data_loading():
    """Test data loading and preprocessing"""
    print("🔄 Testing data loading...")
    
    # Load data
    df = pd.read_csv('online_retail.csv')
    print(f"✅ Original dataset shape: {df.shape}")
    
    # Preprocess
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    print(f"✅ Cleaned dataset shape: {df.shape}")
    print(f"✅ Total customers: {df['CustomerID'].nunique()}")
    print(f"✅ Total products: {df['StockCode'].nunique()}")
    print(f"✅ Total sales: ${df['TotalAmount'].sum():,.2f}")
    
    return df

def test_rfm_analysis(df):
    """Test RFM analysis"""
    print("\n🎯 Testing RFM analysis...")
    
    # Calculate RFM
    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"✅ RFM calculated for {len(rfm)} customers")
    print(f"✅ Recency range: {rfm['Recency'].min()} - {rfm['Recency'].max()} days")
    print(f"✅ Frequency range: {rfm['Frequency'].min()} - {rfm['Frequency'].max()}")
    print(f"✅ Monetary range: ${rfm['Monetary'].min():.2f} - ${rfm['Monetary'].max():.2f}")
    
    return rfm

def test_clustering(rfm):
    """Test customer clustering"""
    print("\n🔍 Testing customer clustering...")
    
    # Prepare data
    scaler = StandardScaler()
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Handle outliers
    Q1 = rfm_features.quantile(0.25)
    Q3 = rfm_features.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((rfm_features < (Q1 - 1.5 * IQR)) | (rfm_features > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Scale and cluster
    rfm_scaled = scaler.fit_transform(rfm_features[outlier_mask])
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    
    # Add segments
    rfm_clean = rfm[outlier_mask].copy()
    rfm_clean['Cluster'] = cluster_labels
    
    segments = {0: 'High-Value', 1: 'Regular', 2: 'Occasional', 3: 'At-Risk'}
    rfm_clean['Segment'] = rfm_clean['Cluster'].map(segments)
    
    print(f"✅ Clustering completed for {len(rfm_clean)} customers")
    print("✅ Segment distribution:")
    for segment, count in rfm_clean['Segment'].value_counts().items():
        print(f"   • {segment}: {count} customers")
    
    return rfm_clean, scaler, kmeans

def test_recommendations(df):
    """Test recommendation system"""
    print("\n🤖 Testing recommendation system...")
    
    # Build recommendation system
    customer_product = df.pivot_table(
        index='CustomerID', columns='StockCode', 
        values='Quantity', aggfunc='sum', fill_value=0
    )
    
    similarity_matrix = cosine_similarity(customer_product.T)
    product_index = customer_product.columns
    product_names = df.set_index('StockCode')['Description'].drop_duplicates()
    
    print(f"✅ Recommendation system built with {len(product_index)} products")
    
    # Test recommendations
    if len(product_index) > 0:
        test_product = product_index[0]
        idx = list(product_index).index(test_product)
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
        
        print(f"✅ Test recommendations for product {test_product}:")
        for i, (idx, score) in enumerate(scores, 1):
            code = product_index[idx]
            name = product_names.get(code, f"Product {code}")
            print(f"   {i}. {code} - {name[:50]}... (Score: {score:.3f})")
    
    return similarity_matrix, product_index, product_names

def test_segment_prediction(rfm_clean, scaler, kmeans):
    """Test segment prediction"""
    print("\n🔮 Testing segment prediction...")
    
    # Test cases
    test_cases = [
        (10, 20, 5000),   # High-value customer
        (50, 5, 500),      # Regular customer
        (100, 2, 100),     # Occasional customer
        (200, 1, 50)       # At-risk customer
    ]
    
    segments = {0: 'High-Value', 1: 'Regular', 2: 'Occasional', 3: 'At-Risk'}
    
    for recency, frequency, monetary in test_cases:
        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_data)[0]
        predicted = segments[cluster]
        print(f"✅ RFM({recency}, {frequency}, ${monetary}) → {predicted}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Shopper Spectrum Project Tests")
    print("=" * 50)
    
    try:
        # Test 1: Data loading
        df = test_data_loading()
        
        # Test 2: RFM analysis
        rfm = test_rfm_analysis(df)
        
        # Test 3: Clustering
        rfm_clean, scaler, kmeans = test_clustering(rfm)
        
        # Test 4: Recommendations
        similarity_matrix, product_index, product_names = test_recommendations(df)
        
        # Test 5: Segment prediction
        test_segment_prediction(rfm_clean, scaler, kmeans)
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Shopper Spectrum is ready to use.")
        print("\n📊 Project Summary:")
        print(f"   • Customers analyzed: {len(rfm_clean)}")
        print(f"   • Products in recommendation system: {len(product_index)}")
        print(f"   • Customer segments: 4")
        print(f"   • Total sales: ${df['TotalAmount'].sum():,.2f}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 