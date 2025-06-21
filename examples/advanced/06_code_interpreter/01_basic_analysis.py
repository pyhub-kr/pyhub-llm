"""Basic data analysis with Code Interpreter tool."""

from pyhub.llm import OpenAILLM, AnthropicLLM, OllamaLLM
from pyhub.llm.agents.tools import CodeInterpreter


def basic_data_analysis():
    """Demonstrate basic data analysis capabilities."""
    
    # Initialize LLM (works with any provider)
    llm = OpenAILLM(model="gpt-4")
    # llm = AnthropicLLM(model="claude-3-sonnet-20240229")
    # llm = OllamaLLM(model="codellama:13b-instruct")
    
    # Create Code Interpreter tool
    code_tool = CodeInterpreter(backend="local")
    
    # Example 1: Basic calculation and statistics
    print("=== Example 1: Basic Statistics ===")
    prompt1 = """
    Create a dataset of 100 random numbers from a normal distribution with mean=50 and std=10.
    Calculate and show:
    1. Basic statistics (mean, median, std, min, max)
    2. Create a histogram
    3. Check if the data follows normal distribution
    """
    
    response1 = llm.ask_with_tools(prompt1, tools=[code_tool])
    print(response1.text)
    print()
    
    # Example 2: Data analysis with pandas
    print("=== Example 2: Sales Data Analysis ===")
    prompt2 = """
    Create a sales dataset with the following:
    - Months: Jan to Dec
    - Sales: [45000, 52000, 48000, 58000, 62000, 69000, 72000, 68000, 66000, 71000, 74000, 78000]
    - Costs: [30000, 34000, 32000, 36000, 38000, 41000, 43000, 40000, 39000, 42000, 44000, 45000]
    
    Analyze:
    1. Calculate profit for each month
    2. Find the month with highest profit
    3. Calculate profit margin percentage
    4. Create a line plot showing sales, costs, and profit trends
    """
    
    response2 = llm.ask_with_tools(prompt2, tools=[code_tool])
    print(response2.text)
    print()
    
    # Example 3: Quick analysis
    print("=== Example 3: Quick Correlation Analysis ===")
    prompt3 = """
    I have this data:
    Study Hours: [1, 2, 3, 4, 5, 6, 7, 8]
    Test Scores: [45, 55, 60, 65, 70, 75, 85, 90]
    
    Calculate the correlation and create a scatter plot with regression line.
    """
    
    response3 = llm.ask_with_tools(prompt3, tools=[code_tool])
    print(response3.text)


if __name__ == "__main__":
    basic_data_analysis()