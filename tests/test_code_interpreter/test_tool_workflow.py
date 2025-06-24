"""Test complete data analysis workflows with CodeInterpreter."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterWorkflow:
    """Test complex multi-step workflows."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_complete_data_analysis_workflow(self, tool):
        """Test a complete data analysis workflow from start to finish."""
        session_id = "complete_workflow"
        
        # Step 1: Data Generation
        result1 = tool.run("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic e-commerce data
n_customers = 1000
n_orders = 5000

# Customer data
customers = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(35, 12, n_customers).astype(int).clip(18, 70),
    'join_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1095, n_customers), 'D'),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
    'tier': np.random.choice(['Bronze', 'Silver', 'Gold'], n_customers, p=[0.6, 0.3, 0.1])
})

print("Customer data created:")
print(customers.head())
print(f"\\nTotal customers: {len(customers)}")
""", session_id=session_id)
        assert "Customer data created:" in result1
        assert "Total customers: 1000" in result1
        
        # Step 2: Create order data
        result2 = tool.run("""
# Generate order data
orders = pd.DataFrame({
    'order_id': range(1, n_orders + 1),
    'customer_id': np.random.choice(customers['customer_id'], n_orders),
    'order_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_orders), 'D'),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_orders),
    'order_value': np.random.lognormal(3.5, 1.2, n_orders).round(2)
})

# Merge with customer data
order_details = orders.merge(customers[['customer_id', 'tier', 'city']], on='customer_id')

print("Order data created and merged:")
print(order_details.head())
print(f"\\nTotal orders: {len(orders)}")
print(f"Date range: {orders['order_date'].min()} to {orders['order_date'].max()}")
""", session_id=session_id)
        assert "Order data created and merged:" in result2
        assert "Total orders: 5000" in result2
        
        # Step 3: Exploratory Data Analysis
        result3 = tool.run("""
# Calculate key metrics
metrics = {
    'Total Revenue': f"${order_details['order_value'].sum():,.2f}",
    'Average Order Value': f"${order_details['order_value'].mean():.2f}",
    'Orders per Customer': f"{len(orders) / len(customers):.1f}",
    'Top Category': order_details['product_category'].value_counts().index[0]
}

print("Key Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value}")

# Customer segmentation
customer_summary = order_details.groupby('customer_id').agg({
    'order_value': ['sum', 'count', 'mean']
}).round(2)
customer_summary.columns = ['total_spent', 'order_count', 'avg_order_value']

print(f"\\nCustomer Summary Statistics:")
print(customer_summary.describe())
""", session_id=session_id)
        assert "Key Metrics:" in result3
        assert "Total Revenue:" in result3
        assert "Customer Summary Statistics:" in result3
        
        # Step 4: Advanced Analysis
        result4 = tool.run("""
# RFM Analysis (Recency, Frequency, Monetary)
latest_date = orders['order_date'].max()

rfm = order_details.groupby('customer_id').agg({
    'order_date': lambda x: (latest_date - x.max()).days,  # Recency
    'order_id': 'count',  # Frequency
    'order_value': 'sum'  # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print("RFM Analysis completed:")
print(rfm.head())

# Identify segments
champion_customers = rfm[rfm['RFM_Score'] == '444'].shape[0]
at_risk_customers = rfm[rfm['RFM_Score'].str.startswith('1')].shape[0]
print(f"\\nChampion customers (444): {champion_customers}")
print(f"At-risk customers (1**): {at_risk_customers}")
""", session_id=session_id)
        assert "RFM Analysis completed:" in result4
        assert "Champion customers" in result4
        
        # Step 5: Visualization Dashboard
        result5 = tool.run("""
# Create comprehensive dashboard
fig = plt.figure(figsize=(16, 12))

# 1. Revenue over time
ax1 = plt.subplot(3, 3, 1)
daily_revenue = order_details.groupby('order_date')['order_value'].sum()
daily_revenue.rolling(7).mean().plot(ax=ax1, color='blue')
ax1.set_title('Daily Revenue (7-day MA)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue ($)')

# 2. Orders by category
ax2 = plt.subplot(3, 3, 2)
category_orders = order_details['product_category'].value_counts()
category_orders.plot(kind='bar', ax=ax2, color='green')
ax2.set_title('Orders by Category')
ax2.set_xlabel('Category')
ax2.set_ylabel('Number of Orders')

# 3. Customer tier distribution
ax3 = plt.subplot(3, 3, 3)
tier_revenue = order_details.groupby('tier')['order_value'].sum()
tier_revenue.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
ax3.set_title('Revenue by Customer Tier')

# 4. City performance
ax4 = plt.subplot(3, 3, 4)
city_stats = order_details.groupby('city')['order_value'].agg(['mean', 'count'])
city_stats['mean'].plot(kind='barh', ax=ax4, color='orange')
ax4.set_title('Average Order Value by City')
ax4.set_xlabel('Average Order Value ($)')

# 5. Order value distribution
ax5 = plt.subplot(3, 3, 5)
order_details['order_value'].hist(bins=50, ax=ax5, color='purple', alpha=0.7)
ax5.set_title('Order Value Distribution')
ax5.set_xlabel('Order Value ($)')
ax5.set_ylabel('Frequency')
ax5.set_xlim(0, 200)

# 6. RFM Heatmap
ax6 = plt.subplot(3, 3, 6)
rfm_pivot = rfm.groupby(['F_Score', 'M_Score']).size().unstack(fill_value=0)
sns.heatmap(rfm_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
ax6.set_title('Customer Count by Frequency-Monetary Scores')

# 7. Monthly trend
ax7 = plt.subplot(3, 3, 7)
order_details['month'] = order_details['order_date'].dt.to_period('M')
monthly_orders = order_details.groupby('month').size()
monthly_orders.plot(ax=ax7, marker='o', color='red')
ax7.set_title('Monthly Order Count')
ax7.set_xlabel('Month')
ax7.set_ylabel('Number of Orders')

# 8. Customer acquisition cohort
ax8 = plt.subplot(3, 3, 8)
customers['join_month'] = customers['join_date'].dt.to_period('M')
cohort_sizes = customers.groupby('join_month').size()
cohort_sizes.plot(kind='bar', ax=ax8, color='teal')
ax8.set_title('Customer Acquisition by Month')
ax8.set_xlabel('Join Month')
ax8.set_ylabel('New Customers')

plt.suptitle('E-commerce Analytics Dashboard', fontsize=16)
plt.tight_layout()
plt.savefig('ecommerce_dashboard.png', dpi=200, bbox_inches='tight')
print("Dashboard created and saved")

# Summary report
print("\\n=== ANALYSIS SUMMARY ===")
print(f"Total Customers: {len(customers)}")
print(f"Total Orders: {len(orders)}")
print(f"Total Revenue: ${order_details['order_value'].sum():,.2f}")
print(f"Customer Lifetime Value: ${order_details['order_value'].sum() / len(customers):.2f}")
print(f"Best Performing Category: {category_orders.index[0]} ({category_orders.iloc[0]} orders)")
print(f"Best Performing City: {city_stats['mean'].idxmax()} (${city_stats['mean'].max():.2f} avg order)")
""", session_id=session_id)
        assert "Dashboard created and saved" in result5
        assert "ANALYSIS SUMMARY" in result5
        assert "ecommerce_dashboard.png" in result5
    
    def test_machine_learning_workflow(self, tool):
        """Test a machine learning workflow with model training and evaluation."""
        session_id = "ml_workflow"
        
        # Step 1: Create dataset
        result1 = tool.run("""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
    flip_y=0.1  # Add 10% noise
)

# Create DataFrame for better handling
feature_names = [f'feature_{i}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Dataset created:")
print(f"Shape: {df.shape}")
print(f"Class distribution:\\n{df['target'].value_counts()}")
print(f"\\nFirst few rows:")
print(df.head())
""", session_id=session_id)
        assert "Dataset created:" in result1
        assert "Shape: (1000, 21)" in result1
        
        # Step 2: Data preparation
        result2 = tool.run("""
# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preparation completed:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\\nFeature scaling applied")
print(f"Mean of scaled features: {X_train_scaled.mean():.6f}")
print(f"Std of scaled features: {X_train_scaled.std():.6f}")
""", session_id=session_id)
        assert "Data preparation completed:" in result2
        assert "Training set: (800, 20)" in result2
        
        # Step 3: Train multiple models
        result3 = tool.run("""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True)  # Enable probability for predict_proba
}

# Train models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy
    }
    print(f"{name} - Accuracy: {accuracy:.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\\nBest model: {best_model_name}")
""", session_id=session_id)
        assert "Accuracy:" in result3
        assert "Best model:" in result3
        
        # Step 4: Model evaluation
        result4 = tool.run("""
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Get predictions from best model
y_pred_best = results[best_model_name]['predictions']
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print(f"Confusion Matrix for {best_model_name}:")
print(cm)

# Classification report
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Feature importance (for Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\\nTop 10 Important Features:")
    print(feature_importance.head(10))
""", session_id=session_id)
        assert "Confusion Matrix" in result4
        assert "Classification Report:" in result4
        
        # Step 5: Visualization
        result5 = tool.run("""
# Create evaluation plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 0].set_xlim([0.0, 1.0])
axes[0, 0].set_ylim([0.0, 1.05])
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend(loc="lower right")

# 2. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. Model Comparison
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
axes[1, 0].bar(model_names, accuracies, color=['blue', 'green', 'red'])
axes[1, 0].set_title('Model Accuracy Comparison')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_ylim([0, 1])
for i, acc in enumerate(accuracies):
    axes[1, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')

# 4. Prediction distribution
axes[1, 1].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                bins=30, label=['Class 0', 'Class 1'], alpha=0.7)
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].legend()

plt.suptitle(f'Model Evaluation - Best Model: {best_model_name}', fontsize=14)
plt.tight_layout()
plt.savefig('ml_evaluation.png', dpi=150, bbox_inches='tight')
print("Evaluation plots saved")

# Final summary
print(f"\\n=== ML WORKFLOW SUMMARY ===")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
""", session_id=session_id)
        assert "Evaluation plots saved" in result5
        assert "ML WORKFLOW SUMMARY" in result5
        assert "ml_evaluation.png" in result5