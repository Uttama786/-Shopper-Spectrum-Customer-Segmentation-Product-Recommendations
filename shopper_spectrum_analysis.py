import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ShopperSpectrum:
    def __init__(self, data_path):
        """Initialize Shopper Spectrum with dataset path"""
        self.data_path = data_path
        self.df = None
        self.rfm_df = None
        self.product_similarity_matrix = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the retail dataset"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Display initial info
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        # Data cleaning
        # Remove rows with missing CustomerID
        self.df = self.df.dropna(subset=['CustomerID'])
        
        # Remove canceled invoices (starting with 'C')
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Remove negative or zero quantities and prices
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        
        # Convert InvoiceDate to datetime
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Calculate total amount
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        print(f"✅ Cleaned dataset shape: {self.df.shape}")
        print(f"📅 Date range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        
        return self.df
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("\n📈 Performing Exploratory Data Analysis...")
        
        # Create EDA visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Country-wise sales
        country_sales = self.df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
        axes[0, 0].bar(range(len(country_sales)), country_sales.values, color='skyblue')
        axes[0, 0].set_title('Top 10 Countries by Sales', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Country')
        axes[0, 0].set_ylabel('Total Sales Amount')
        axes[0, 0].set_xticks(range(len(country_sales)))
        axes[0, 0].set_xticklabels(country_sales.index, rotation=45)
        
        # 2. Top selling products
        top_products = self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        axes[0, 1].bar(range(len(top_products)), top_products.values, color='lightcoral')
        axes[0, 1].set_title('Top 10 Products by Quantity Sold', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Product')
        axes[0, 1].set_ylabel('Quantity Sold')
        axes[0, 1].set_xticks(range(len(top_products)))
        axes[0, 1].set_xticklabels([f"Product {i+1}" for i in range(len(top_products))], rotation=45)
        
        # 3. Monthly sales trend
        monthly_sales = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
        axes[1, 0].plot(range(len(monthly_sales)), monthly_sales.values, marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Total Sales Amount')
        axes[1, 0].set_xticks(range(len(monthly_sales)))
        axes[1, 0].set_xticklabels([str(period) for period in monthly_sales.index], rotation=45)
        
        # 4. Customer distribution by purchase frequency
        customer_freq = self.df.groupby('CustomerID')['InvoiceNo'].nunique()
        axes[1, 1].hist(customer_freq.values, bins=30, color='gold', alpha=0.7)
        axes[1, 1].set_title('Distribution of Purchase Frequency', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Purchases')
        axes[1, 1].set_ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key statistics
        print(f"📊 Total customers: {self.df['CustomerID'].nunique()}")
        print(f"📦 Total products: {self.df['StockCode'].nunique()}")
        print(f"💰 Total sales: ${self.df['TotalAmount'].sum():,.2f}")
        print(f"🌍 Countries: {self.df['Country'].nunique()}")
    
    def calculate_rfm(self):
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        print("\n🎯 Calculating RFM metrics...")
        
        # Set reference date (last date + 1 day)
        reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        
        # Calculate RFM for each customer
        rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Display RFM statistics
        print("📊 RFM Statistics:")
        print(rfm.describe())
        
        self.rfm_df = rfm
        return rfm
    
    def perform_clustering(self):
        """Perform K-means clustering on RFM data"""
        print("\n🔍 Performing customer segmentation...")
        
        # Prepare data for clustering
        rfm_data = self.rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

        # Handle outliers using IQR method (before scaling)
        Q1 = rfm_data.quantile(0.25)
        Q3 = rfm_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ~((rfm_data < (Q1 - 1.5 * IQR)) | (rfm_data > (Q3 + 1.5 * IQR))).any(axis=1)
        rfm_filtered = rfm_data[outlier_mask]

        # Scale the filtered data
        rfm_scaled = self.scaler.fit_transform(rfm_filtered)

        # Find optimal number of clusters using Elbow method
        inertias = []
        silhouette_scores = []
        K_range = range(2, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

        # Plot Elbow method
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)

        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Choose optimal k (usually 4 for RFM)
        optimal_k = 4
        print(f"🎯 Optimal number of clusters: {optimal_k}")

        # Perform final clustering
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(rfm_scaled)

        # Add cluster labels to RFM data
        rfm_with_clusters = self.rfm_df[outlier_mask].copy()
        rfm_with_clusters['Cluster'] = cluster_labels

        # Analyze clusters
        cluster_analysis = rfm_with_clusters.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)

        cluster_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
        print("\n📊 Cluster Analysis:")
        print(cluster_analysis)

        # Visualize clusters
        fig = plt.figure(figsize=(15, 5))

        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(rfm_with_clusters['Recency'], 
                             rfm_with_clusters['Frequency'], 
                             rfm_with_clusters['Monetary'],
                             c=cluster_labels, cmap='viridis')
        ax1.set_xlabel('Recency')
        ax1.set_ylabel('Frequency')
        ax1.set_zlabel('Monetary')
        ax1.set_title('3D Cluster Visualization')

        # 2D plots
        ax2 = fig.add_subplot(132)
        scatter = ax2.scatter(rfm_with_clusters['Recency'], 
                             rfm_with_clusters['Frequency'],
                             c=cluster_labels, cmap='viridis')
        ax2.set_xlabel('Recency')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Recency vs Frequency')

        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(rfm_with_clusters['Frequency'], 
                             rfm_with_clusters['Monetary'],
                             c=cluster_labels, cmap='viridis')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Monetary')
        ax3.set_title('Frequency vs Monetary')

        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Assign segment names
        segment_names = {
            0: 'High-Value Customers',
            1: 'Regular Customers', 
            2: 'Occasional Customers',
            3: 'At-Risk Customers'
        }

        rfm_with_clusters['Segment'] = rfm_with_clusters['Cluster'].map(segment_names)

        self.rfm_df = rfm_with_clusters
        return rfm_with_clusters
    
    def build_recommendation_system(self):
        """Build item-based collaborative filtering recommendation system"""
        print("\n🤖 Building product recommendation system...")
        
        # Create customer-product matrix
        customer_product_matrix = self.df.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )
        
        # Calculate product similarity matrix
        self.product_similarity_matrix = cosine_similarity(customer_product_matrix.T)
        
        # Create product index mapping
        self.product_index = customer_product_matrix.columns
        self.product_names = self.df.set_index('StockCode')['Description'].drop_duplicates()
        
        print(f"✅ Recommendation system built with {len(self.product_index)} products")
        return self.product_similarity_matrix
    
    def get_product_recommendations(self, product_code, n_recommendations=5):
        """Get product recommendations based on product code"""
        if product_code not in self.product_index:
            return []
        
        # Get product index
        product_idx = list(self.product_index).index(product_code)
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.product_similarity_matrix[product_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar products (excluding the product itself)
        similar_products = similarity_scores[1:n_recommendations+1]
        
        recommendations = []
        for idx, score in similar_products:
            product_code = self.product_index[idx]
            product_name = self.product_names.get(product_code, f"Product {product_code}")
            recommendations.append({
                'ProductCode': product_code,
                'ProductName': product_name,
                'SimilarityScore': round(score, 3)
            })
        
        return recommendations
    
    def predict_customer_segment(self, recency, frequency, monetary):
        """Predict customer segment based on RFM values"""
        # Scale the input values
        input_data = np.array([[recency, frequency, monetary]])
        scaled_input = self.scaler.transform(input_data)
        
        # Predict cluster
        cluster = self.kmeans_model.predict(scaled_input)[0]
        
        # Map to segment name
        segment_names = {
            0: 'High-Value Customers',
            1: 'Regular Customers',
            2: 'Occasional Customers', 
            3: 'At-Risk Customers'
        }
        
        return segment_names[cluster]
    
    def run_complete_analysis(self):
        """Run the complete Shopper Spectrum analysis"""
        print("🚀 Starting Shopper Spectrum Analysis...")
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Perform EDA
        self.perform_eda()
        
        # Step 3: Calculate RFM
        self.calculate_rfm()
        
        # Step 4: Perform clustering
        self.perform_clustering()
        
        # Step 5: Build recommendation system
        self.build_recommendation_system()
        
        print("\n✅ Shopper Spectrum analysis completed!")
        print("📊 Key Results:")
        print(f"   • Total customers analyzed: {len(self.rfm_df)}")
        print(f"   • Customer segments identified: {self.rfm_df['Segment'].nunique()}")
        print(f"   • Products in recommendation system: {len(self.product_index)}")
        
        return self

# Run the analysis
if __name__ == "__main__":
    shopper = ShopperSpectrum('online_retail.csv')
    shopper.run_complete_analysis() 