"""Test pandas functionality with CodeInterpreter in multi-step execution."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterPandas:
    """Test pandas operations across multiple runs."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_dataframe_creation_and_manipulation(self, tool):
        """Test creating and manipulating DataFrames step by step."""
        session_id = "pandas_basic"
        
        # Step 1: Create DataFrame
        result1 = tool.run("""
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 75000, 90000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
})
print("DataFrame created:")
print(df)
print(f"\\nShape: {df.shape}")
""", session_id=session_id)
        assert "DataFrame created:" in result1
        assert "Shape: (5, 4)" in result1
        
        # Step 2: Add calculated column
        result2 = tool.run("""
# Add bonus column (10% of salary)
df['bonus'] = df['salary'] * 0.1
print("After adding bonus column:")
print(df)
""", session_id=session_id)
        assert "bonus" in result2
        assert "5000" in result2  # 10% of 50000
        
        # Step 3: Filter data
        result3 = tool.run("""
# Filter IT department
it_employees = df[df['department'] == 'IT']
print("IT Department employees:")
print(it_employees)
print(f"\\nAverage IT salary: ${it_employees['salary'].mean():,.2f}")
""", session_id=session_id)
        assert "IT Department employees:" in result3
        assert "Alice" in result3
        assert "Charlie" in result3
        assert "$62,500.00" in result3  # Average of 50000 and 75000
    
    def test_data_aggregation_pipeline(self, tool):
        """Test progressive data aggregation."""
        session_id = "pandas_agg"
        
        # Step 1: Create sales data
        result1 = tool.run("""
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
sales_data = pd.DataFrame({
    'date': dates,
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['East', 'West', 'North', 'South'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'units': np.random.randint(1, 50, 100)
})
print(f"Sales data created: {len(sales_data)} records")
print(sales_data.head())
""", session_id=session_id)
        assert "100 records" in result1
        
        # Step 2: Basic aggregation
        result2 = tool.run("""
# Group by product
product_summary = sales_data.groupby('product').agg({
    'sales': ['sum', 'mean', 'count'],
    'units': 'sum'
})
print("Product Summary:")
print(product_summary)
""", session_id=session_id)
        assert "Product Summary:" in result2
        assert "sum" in result2
        assert "mean" in result2
        
        # Step 3: Time-based analysis
        result3 = tool.run("""
# Add month column
sales_data['month'] = sales_data['date'].dt.month_name()
monthly_sales = sales_data.groupby('month')['sales'].sum().sort_values(ascending=False)
print("Monthly Sales (Top 5):")
print(monthly_sales.head())
""", session_id=session_id)
        assert "Monthly Sales" in result3
        
        # Step 4: Pivot table
        result4 = tool.run("""
# Create pivot table
pivot = pd.pivot_table(
    sales_data,
    values='sales',
    index='product',
    columns='region',
    aggfunc='sum',
    fill_value=0
)
print("Sales by Product and Region:")
print(pivot)
print(f"\\nTotal sales: ${sales_data['sales'].sum():,}")
""", session_id=session_id)
        assert "Sales by Product and Region:" in result4
        assert "East" in result4
        assert "Total sales:" in result4
    
    def test_data_cleaning_workflow(self, tool):
        """Test data cleaning in multiple steps."""
        session_id = "pandas_clean"
        
        # Step 1: Create messy data
        result1 = tool.run("""
import pandas as pd
import numpy as np

# Create data with issues
df = pd.DataFrame({
    'id': range(1, 11),
    'value': [10, np.nan, 30, 40, np.nan, 60, 70, 80, np.nan, 100],
    'category': ['A', 'B', 'A', None, 'B', 'C', 'A', None, 'C', 'B'],
    'price': ['$10.50', '$20.00', 'N/A', '$30.75', '$40.00', 
              '$50.25', 'Missing', '$70.00', '$80.50', '$90.00']
})
print("Original data with issues:")
print(df)
print(f"\\nMissing values:\\n{df.isnull().sum()}")
""", session_id=session_id)
        assert "Original data with issues:" in result1
        assert "NaN" in result1
        assert "Missing values:" in result1
        
        # Step 2: Handle missing values
        result2 = tool.run("""
# Fill missing values in 'value' column with mean
df['value'].fillna(df['value'].mean(), inplace=True)

# Fill missing categories with 'Unknown'
df['category'].fillna('Unknown', inplace=True)

print("After handling missing values:")
print(df)
print(f"\\nRemaining missing values:\\n{df.isnull().sum()}")
""", session_id=session_id)
        assert "After handling missing values:" in result2
        assert "Unknown" in result2
        
        # Step 3: Clean price column
        result3 = tool.run("""
# Clean price column
df['price_clean'] = df['price'].replace(['N/A', 'Missing'], '0')
df['price_clean'] = df['price_clean'].str.replace('$', '').astype(float)

print("After cleaning price:")
print(df[['price', 'price_clean']])
print(f"\\nAverage price: ${df['price_clean'].mean():.2f}")
""", session_id=session_id)
        assert "After cleaning price:" in result3
        assert "price_clean" in result3
        assert "Average price:" in result3
        
        # Step 4: Final summary
        result4 = tool.run("""
# Summary statistics
summary = df.describe()
print("Clean data summary:")
print(summary)

# Category counts
print("\\nCategory distribution:")
print(df['category'].value_counts())
""", session_id=session_id)
        assert "Clean data summary:" in result4
        assert "Category distribution:" in result4
    
    def test_merge_and_join_operations(self, tool):
        """Test merging DataFrames progressively."""
        session_id = "pandas_merge"
        
        # Step 1: Create first DataFrame
        result1 = tool.run("""
import pandas as pd

# Customer data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
})
print("Customers DataFrame:")
print(customers)
""", session_id=session_id)
        assert "Customers DataFrame:" in result1
        
        # Step 2: Create second DataFrame
        result2 = tool.run("""
# Orders data
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 1, 3, 4, 2],
    'amount': [100, 200, 150, 300, 250, 180]
})
print("\\nOrders DataFrame:")
print(orders)
""", session_id=session_id)
        assert "Orders DataFrame:" in result2
        
        # Step 3: Merge DataFrames
        result3 = tool.run("""
# Merge customers with orders
merged = pd.merge(customers, orders, on='customer_id', how='left')
print("\\nMerged data:")
print(merged)

# Calculate total per customer
customer_totals = merged.groupby('name')['amount'].sum().fillna(0)
print("\\nTotal amount per customer:")
print(customer_totals)
""", session_id=session_id)
        assert "Merged data:" in result3
        assert "Total amount per customer:" in result3
        assert "Alice" in result3
    
    def test_pandas_with_datetime(self, tool):
        """Test datetime operations in pandas."""
        session_id = "pandas_datetime"
        
        # Step 1: Create time series data
        result1 = tool.run("""
import pandas as pd
import numpy as np

# Create date range
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(1000, 5000, len(dates)),
    'temperature': np.random.uniform(0, 30, len(dates))
})
df['day_of_week'] = df['date'].dt.day_name()
print("Time series data:")
print(df.head())
print(f"\\nDate range: {df['date'].min()} to {df['date'].max()}")
""", session_id=session_id)
        assert "Time series data:" in result1
        assert "2024-01-01" in result1
        assert "2024-01-31" in result1
        
        # Step 2: Resample to weekly
        result2 = tool.run("""
# Set date as index
df.set_index('date', inplace=True)

# Resample to weekly
weekly = df.resample('W').agg({
    'sales': 'sum',
    'temperature': 'mean'
})
print("\\nWeekly aggregation:")
print(weekly)
""", session_id=session_id)
        assert "Weekly aggregation:" in result2
        
        # Step 3: Day of week analysis
        result3 = tool.run("""
# Reset index for groupby
df_reset = df.reset_index()

# Group by day of week
dow_sales = df_reset.groupby('day_of_week')['sales'].mean().sort_values(ascending=False)
print("\\nAverage sales by day of week:")
print(dow_sales)
""", session_id=session_id)
        assert "Average sales by day of week:" in result3
        assert "Monday" in result3 or "Sunday" in result3