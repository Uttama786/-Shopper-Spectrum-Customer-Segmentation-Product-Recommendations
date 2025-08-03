# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

## 📋 **Project Overview**

**Shopper Spectrum** is a comprehensive machine learning project that analyzes e-commerce customer behavior to perform customer segmentation and provide product recommendations. The project uses RFM (Recency, Frequency, Monetary) analysis and collaborative filtering to deliver actionable business insights.

## 🎯 **Key Features**

### **Customer Segmentation**
• **RFM Analysis**: Recency, Frequency, Monetary metrics calculation
• **K-Means Clustering**: Automated customer segmentation into 4 segments
• **Segment Analysis**: High-Value, Regular, Occasional, and At-Risk customers
• **Interactive Predictor**: Predict customer segments based on RFM values

### **Product Recommendations**
• **Collaborative Filtering**: Item-based recommendation system
• **Cosine Similarity**: Product similarity calculations
• **Top 5 Recommendations**: Similar products for any given product
• **Product Analytics**: Top-selling products and trends

### **Data Analysis**
• **Exploratory Data Analysis**: Comprehensive visualizations
• **Sales Analytics**: Country-wise and product-wise analysis
• **Trend Analysis**: Monthly sales patterns and customer behavior
• **Interactive Dashboard**: Real-time metrics and insights

## 🚀 **Quick Start**

### **Installation**
```bash
pip install -r requirements.txt
```

### **Run Analysis**
```bash
python shopper_spectrum_analysis.py
```

### **Launch Streamlit App**
```bash
streamlit run app.py
```

## 📊 **Project Structure**

```
Shopper Spectrum/
├── online_retail.csv          # Dataset
├── shopper_spectrum_analysis.py  # Main analysis script
├── app.py                     # Streamlit web application
├── requirements.txt           # Dependencies
├── README.md                 # Project documentation
└── Generated Files/
    ├── eda_visualizations.png
    ├── clustering_analysis.png
    └── cluster_visualization.png
```

## 🔧 **Technical Implementation**

### **Data Preprocessing**
• Remove missing CustomerIDs
• Filter out canceled invoices
• Remove negative/zero quantities and prices
• Calculate total transaction amounts

### **RFM Analysis**
• **Recency**: Days since last purchase
• **Frequency**: Number of purchases
• **Monetary**: Total amount spent

### **Clustering Algorithm**
• **K-Means**: 4 customer segments
• **StandardScaler**: Feature normalization
• **Elbow Method**: Optimal cluster selection
• **Silhouette Score**: Cluster quality validation

### **Recommendation System**
• **Customer-Product Matrix**: Purchase history matrix
• **Cosine Similarity**: Product similarity calculation
• **Item-Based Filtering**: Collaborative filtering approach

## 📈 **Customer Segments**

| Segment | Characteristics | Strategy |
|---------|----------------|----------|
| **High-Value** | Recent, frequent, high spending | VIP treatment, exclusive offers |
| **Regular** | Moderate activity, consistent | Loyalty programs, regular promotions |
| **Occasional** | Infrequent buyers, low engagement | Re-engagement campaigns |
| **At-Risk** | Haven't purchased recently | Win-back campaigns, special offers |

## 🎨 **Visualizations**

### **Dashboard Features**
• **Key Metrics**: Customer count, product count, total sales, countries
• **Sales Analytics**: Country-wise sales distribution
• **Segment Distribution**: Customer segment pie chart
• **Trend Analysis**: Monthly sales patterns

### **Segmentation Module**
• **RFM Distribution**: Histograms for each metric by segment
• **Cluster Analysis**: 3D and 2D cluster visualizations
• **Segment Statistics**: Average RFM values by segment
• **Interactive Predictor**: Real-time segment prediction

### **Recommendations Module**
• **Product Similarity**: Top 5 similar products
• **Product Analytics**: Top-selling products chart
• **Similarity Scores**: Detailed recommendation scores

## 🛠 **Technologies Used**

• **Python**: Core programming language
• **Pandas**: Data manipulation and analysis
• **NumPy**: Numerical computations
• **Scikit-learn**: Machine learning algorithms
• **Matplotlib/Seaborn**: Static visualizations
• **Plotly**: Interactive visualizations
• **Streamlit**: Web application framework

## 📊 **Dataset Information**

**Source**: Online retail transaction dataset
**Features**:
• InvoiceNo: Transaction identifier
• StockCode: Product code
• Description: Product description
• Quantity: Units purchased
• InvoiceDate: Transaction date
• UnitPrice: Price per unit
• CustomerID: Customer identifier
• Country: Customer location
drive link:-https://drive.google.com/file/d/1ou6Dc5gsXyT16NUUCRW7wQr-3i0fuiuB/view?usp=sharing

## 🎯 **Business Applications**

### **Marketing Strategy**
• **Targeted Campaigns**: Segment-specific marketing
• **Personalization**: Customized product recommendations
• **Customer Retention**: At-risk customer identification
• **Revenue Optimization**: High-value customer focus

### **Inventory Management**
• **Demand Forecasting**: Based on customer segments
• **Product Placement**: Strategic product positioning
• **Pricing Strategy**: Segment-based pricing

### **Customer Service**
• **VIP Treatment**: High-value customer prioritization
• **Proactive Support**: At-risk customer outreach
• **Personalized Experience**: Segment-specific service

## 📈 **Performance Metrics**

• **Customer Segmentation Accuracy**: Silhouette score analysis
• **Recommendation Quality**: Similarity score validation
• **Business Impact**: Segment-specific revenue analysis
• **User Engagement**: Interactive dashboard usage

## 🔮 **Future Enhancements**

• **Real-time Processing**: Live data integration
• **Advanced ML Models**: Deep learning for recommendations
• **A/B Testing**: Recommendation system optimization
• **Mobile App**: Cross-platform accessibility
• **API Integration**: Third-party system connectivity

## 📝 **Usage Examples**

### **Customer Segmentation**
```python
# Predict customer segment
segment = shopper.predict_customer_segment(
    recency=30,    # days since last purchase
    frequency=5,    # number of purchases
    monetary=100.0  # total amount spent
)
print(f"Customer Segment: {segment}")
```

### **Product Recommendations**
```python
# Get product recommendations
recommendations = shopper.get_product_recommendations(
    product_code="85123A",
    n_recommendations=5
)
for rec in recommendations:
    print(f"{rec['ProductName']} (Score: {rec['SimilarityScore']})")
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

• Dataset: Online retail transaction data
• Libraries: Open-source Python community
• Inspiration: E-commerce analytics best practices

---


**🎉 Ready to transform your e-commerce analytics with Shopper Spectrum!** 
