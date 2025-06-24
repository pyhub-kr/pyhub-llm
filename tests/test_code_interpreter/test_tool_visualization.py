"""Test visualization capabilities with matplotlib and seaborn in multi-step execution."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterVisualization:
    """Test matplotlib and seaborn visualization across multiple runs."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_matplotlib_basic_plots(self, tool):
        """Test creating basic matplotlib plots step by step."""
        session_id = "matplotlib_basic"
        
        # Step 1: Setup and create figure
        result1 = tool.run("""
import matplotlib.pyplot as plt
import numpy as np

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Basic Plot Types', fontsize=16)
print("Figure created with 2x2 subplots")
""", session_id=session_id)
        assert "Figure created with 2x2 subplots" in result1
        
        # Step 2: Add line plot
        result2 = tool.run("""
# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
axes[0, 0].plot(x, y, 'b-', label='sin(x)')
axes[0, 0].set_title('Line Plot')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid(True)
print("Line plot added")
""", session_id=session_id)
        assert "Line plot added" in result2
        
        # Step 3: Add bar plot
        result3 = tool.run("""
# Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[0, 1].bar(categories, values, color='green', alpha=0.7)
axes[0, 1].set_title('Bar Plot')
axes[0, 1].set_ylabel('Values')
print("Bar plot added")
""", session_id=session_id)
        assert "Bar plot added" in result3
        
        # Step 4: Add scatter plot
        result4 = tool.run("""
# Scatter plot
np.random.seed(42)
x_scatter = np.random.randn(50)
y_scatter = 2 * x_scatter + np.random.randn(50) * 0.5
axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6, c='red')
axes[1, 0].set_title('Scatter Plot')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
print("Scatter plot added")
""", session_id=session_id)
        assert "Scatter plot added" in result4
        
        # Step 5: Add histogram
        result5 = tool.run("""
# Histogram
data = np.random.normal(100, 15, 1000)
axes[1, 1].hist(data, bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Histogram')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')

# Adjust layout and save
plt.tight_layout()
plt.savefig('basic_plots.png', dpi=150)
print("All plots completed and saved")
""", session_id=session_id)
        assert "All plots completed and saved" in result5
        assert "basic_plots.png" in result5
    
    def test_seaborn_statistical_plots(self, tool):
        """Test seaborn statistical visualizations."""
        session_id = "seaborn_stats"
        
        # Step 1: Import and create data
        result1 = tool.run("""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value': np.random.gamma(2, 2, 1000)
})
print("Data created for seaborn plots")
print(df.head())
""", session_id=session_id)
        assert "Data created for seaborn plots" in result1
        
        # Step 2: Create distribution plot
        result2 = tool.run("""
# Create figure for distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution plot
sns.histplot(data=df, x='x', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution Plot with KDE')
print("Distribution plot created")
""", session_id=session_id)
        assert "Distribution plot created" in result2
        
        # Step 3: Add box plot
        result3 = tool.run("""
# Box plot by category
sns.boxplot(data=df, x='category', y='value', ax=axes[0, 1])
axes[0, 1].set_title('Box Plot by Category')
print("Box plot added")
""", session_id=session_id)
        assert "Box plot added" in result3
        
        # Step 4: Add violin plot
        result4 = tool.run("""
# Violin plot
sns.violinplot(data=df, x='category', y='y', ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot')
print("Violin plot added")
""", session_id=session_id)
        assert "Violin plot added" in result4
        
        # Step 5: Add regression plot
        result5 = tool.run("""
# Regression plot
sns.regplot(data=df, x='x', y='y', ax=axes[1, 1], scatter_kws={'alpha': 0.5})
axes[1, 1].set_title('Regression Plot')

plt.tight_layout()
plt.savefig('seaborn_stats.png', dpi=150)
print("Seaborn statistical plots completed")
""", session_id=session_id)
        assert "Seaborn statistical plots completed" in result5
    
    def test_complex_visualization_workflow(self, tool):
        """Test building a complex visualization dashboard."""
        session_id = "complex_viz"
        
        # Step 1: Generate complex dataset
        result1 = tool.run("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate time series data
dates = pd.date_range('2023-01-01', periods=365)
df = pd.DataFrame({
    'date': dates,
    'sales': 1000 + np.cumsum(np.random.randn(365) * 10) + np.sin(np.arange(365) * 2 * np.pi / 365) * 200,
    'customers': np.random.poisson(50, 365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'product': np.random.choice(['A', 'B', 'C'], 365)
})
df['month'] = df['date'].dt.month_name()
df['weekday'] = df['date'].dt.day_name()
print(f"Generated dataset with {len(df)} records")
print(df.head())
""", session_id=session_id)
        assert "Generated dataset with 365 records" in result1
        
        # Step 2: Create dashboard layout
        result2 = tool.run("""
# Create dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main time series plot
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['date'], df['sales'], label='Daily Sales', alpha=0.7)
ax1.set_title('Sales Trend Over Time', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales')
ax1.grid(True, alpha=0.3)

# Add 30-day moving average
df['sales_ma30'] = df['sales'].rolling(30).mean()
ax1.plot(df['date'], df['sales_ma30'], 'r-', label='30-day MA', linewidth=2)
ax1.legend()
print("Time series plot added")
""", session_id=session_id)
        assert "Time series plot added" in result2
        
        # Step 3: Add regional analysis
        result3 = tool.run("""
# Regional sales distribution
ax2 = fig.add_subplot(gs[1, 0])
regional_sales = df.groupby('region')['sales'].sum().sort_values(ascending=True)
regional_sales.plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_title('Total Sales by Region')
ax2.set_xlabel('Total Sales')

# Product mix pie chart
ax3 = fig.add_subplot(gs[1, 1])
product_sales = df.groupby('product')['sales'].sum()
ax3.pie(product_sales, labels=product_sales.index, autopct='%1.1f%%', startangle=90)
ax3.set_title('Sales Distribution by Product')
print("Regional and product analysis added")
""", session_id=session_id)
        assert "Regional and product analysis added" in result3
        
        # Step 4: Add correlation heatmap
        result4 = tool.run("""
# Prepare correlation data
ax4 = fig.add_subplot(gs[1, 2])
# Create numeric features for correlation
df['day_of_year'] = df['date'].dt.dayofyear
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
corr_data = df[['sales', 'customers', 'day_of_year', 'is_weekend']].corr()

# Heatmap
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax4)
ax4.set_title('Feature Correlation')
print("Correlation heatmap added")
""", session_id=session_id)
        assert "Correlation heatmap added" in result4
        
        # Step 5: Add weekly patterns
        result5 = tool.run("""
# Weekly pattern analysis
ax5 = fig.add_subplot(gs[2, :2])
weekly_pattern = df.groupby('weekday')['sales'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
weekly_pattern.plot(kind='bar', ax=ax5, color='lightgreen', edgecolor='black')
ax5.set_title('Average Sales by Day of Week')
ax5.set_xlabel('Day of Week')
ax5.set_ylabel('Average Sales')
ax5.tick_params(axis='x', rotation=45)

# Monthly boxplot
ax6 = fig.add_subplot(gs[2, 2])
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
df_sorted = df.sort_values('month')
sns.boxplot(data=df_sorted, x='month', y='customers', ax=ax6)
ax6.set_title('Customer Distribution by Month')
ax6.tick_params(axis='x', rotation=90)

plt.suptitle('Sales Dashboard - 2023', fontsize=16, y=0.98)
plt.savefig('sales_dashboard.png', dpi=200, bbox_inches='tight')
print("Complete dashboard saved")
""", session_id=session_id)
        assert "Complete dashboard saved" in result5
        assert "sales_dashboard.png" in result5
    
    def test_plot_customization_workflow(self, tool):
        """Test progressive plot customization."""
        session_id = "plot_custom"
        
        # Step 1: Create basic plot
        result1 = tool.run("""
import matplotlib.pyplot as plt
import numpy as np

# Create figure and basic plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
line1, = ax.plot(x, np.sin(x), label='Original')
print("Basic plot created")
""", session_id=session_id)
        assert "Basic plot created" in result1
        
        # Step 2: Add more data series
        result2 = tool.run("""
# Add more lines
line2, = ax.plot(x, np.sin(x + np.pi/4), label='Phase π/4', linestyle='--')
line3, = ax.plot(x, np.sin(x + np.pi/2), label='Phase π/2', linestyle=':')
ax.legend()
print("Additional data series added")
""", session_id=session_id)
        assert "Additional data series added" in result2
        
        # Step 3: Customize appearance
        result3 = tool.run("""
# Customize colors and styles
line1.set_color('blue')
line1.set_linewidth(2)
line2.set_color('red')
line2.set_alpha(0.7)
line3.set_color('green')
line3.set_marker('o')
line3.set_markevery(10)

# Add grid and labels
ax.grid(True, alpha=0.3, linestyle='-.')
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)
ax.set_title('Progressively Customized Plot', fontsize=14, fontweight='bold')
print("Styling applied")
""", session_id=session_id)
        assert "Styling applied" in result3
        
        # Step 4: Add annotations
        result4 = tool.run("""
# Add annotations
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.2),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=12, ha='center')

# Add shaded region
ax.axvspan(2, 4, alpha=0.2, color='yellow', label='Region of Interest')

# Update legend
ax.legend(loc='upper right')

# Save the customized plot
plt.savefig('customized_plot.png', dpi=150, bbox_inches='tight')
print("Plot customization complete")
""", session_id=session_id)
        assert "Plot customization complete" in result4
        assert "customized_plot.png" in result4