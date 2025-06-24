"""Data visualization examples with Code Interpreter."""

from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter


def visualization_examples():
    """Demonstrate various visualization capabilities."""
    
    # Initialize components
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    
    # Example 1: Multiple plot types
    print("=== Example 1: Comprehensive Visualization ===")
    prompt1 = """
    Create a comprehensive visualization dashboard with 4 subplots:
    1. Bar chart: Monthly revenue data for a store (generate realistic data)
    2. Pie chart: Market share of top 5 tech companies
    3. Heatmap: Correlation matrix of stock returns (generate sample data)
    4. Box plot: Distribution of employee salaries by department
    
    Make it professional with proper titles, labels, and colors.
    Save as 'dashboard.png' with high DPI.
    """
    
    response1 = llm.ask(prompt1, tools=[code_tool])
    print(response1.text)
    print()
    
    # Example 2: Time series visualization
    print("=== Example 2: Time Series Analysis ===")
    prompt2 = """
    Generate 2 years of daily stock price data with:
    - Upward trend
    - Some volatility
    - Occasional dips
    
    Create visualizations showing:
    1. Price over time with 30-day moving average
    2. Daily returns distribution
    3. Volatility (rolling 30-day standard deviation)
    
    Use seaborn style for better aesthetics.
    """
    
    response2 = llm.ask(prompt2, tools=[code_tool])
    print(response2.text)
    print()
    
    # Example 3: Statistical visualization
    print("=== Example 3: Statistical Plots ===")
    prompt3 = """
    Create a dataset comparing test scores across 3 different teaching methods.
    Each method should have 30 students with slightly different score distributions.
    
    Visualize using:
    1. Violin plots to show distribution
    2. Statistical significance test
    3. Effect size calculation
    
    Add proper statistical annotations.
    """
    
    response3 = llm.ask(prompt3, tools=[code_tool])
    print(response3.text)


def interactive_visualization():
    """Example of creating interactive visualizations."""
    
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    
    print("=== Interactive Visualization (if plotly available) ===")
    prompt = """
    Try to create an interactive visualization using plotly:
    - 3D scatter plot of random data points
    - Color-coded by cluster
    - With hover information
    
    If plotly is not available, create a similar static plot with matplotlib.
    """
    
    response = llm.ask(prompt, tools=[code_tool])
    print(response.text)


if __name__ == "__main__":
    visualization_examples()
    print("\n" + "="*50 + "\n")
    interactive_visualization()