"""Multi-tool usage with Code Interpreter."""

from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter, Calculator


def multi_tool_example():
    """Demonstrate using Code Interpreter with other tools."""
    
    # Initialize components
    llm = OpenAILLM(model="gpt-4")
    
    # Create multiple tools
    code_tool = CodeInterpreter(backend="local")
    calc_tool = Calculator()
    
    # Example 1: Combined analysis
    print("=== Example 1: Financial Analysis with Multiple Tools ===")
    prompt1 = """
    I need to analyze an investment scenario:
    
    1. First, use the calculator to find the compound interest on $10,000 
       at 7% annual rate for 5 years
    
    2. Then, use Python to:
       - Create a visualization showing investment growth over 20 years
       - Compare different interest rates (5%, 7%, 10%)
       - Calculate the time to double the investment for each rate
       - Create a comprehensive investment report
    """
    
    response1 = llm.ask(
        prompt1,
        tools=[calc_tool, code_tool]
    )
    print(response1.text)
    print()
    
    # Example 2: Data validation
    print("=== Example 2: Data Validation and Analysis ===")
    prompt2 = """
    1. Use the calculator to verify: 15% of 2500
    
    2. Then create a sales commission analysis:
       - Sales amounts: [2500, 3200, 1800, 4500, 3900]
       - Commission rate: 15%
       - Calculate commissions
       - Visualize sales vs commissions
       - Find total payout needed
    """
    
    response2 = llm.ask(
        prompt2,
        tools=[calc_tool, code_tool]
    )
    print(response2.text)


def code_interpreter_with_context():
    """Show how Code Interpreter can work with conversation context."""
    
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    
    print("\n=== Code Interpreter with Conversation Context ===")
    
    # First message - establish context
    llm.ask("I'm analyzing customer churn data for a telecom company.")
    
    # Now use code interpreter with context
    response = llm.ask(
        """
        Based on our discussion about telecom customer churn, create a sample dataset with:
        - 500 customers
        - Features: monthly_charges, total_charges, tenure_months, num_services, has_churned
        - Make it realistic for a telecom company
        - Show churn rate and create visualizations to identify patterns
        """,
        tools=[code_tool]
    )
    print(response.text)


def practical_use_case():
    """Practical example: CSV analysis workflow."""
    
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    
    print("\n=== Practical Use Case: Sales Report Analysis ===")
    
    # Simulate having a CSV file
    prompt = """
    Create a sample sales CSV file with:
    - Columns: date, product, category, quantity, unit_price, region
    - 200 rows of realistic data
    - Save as 'sales_data.csv'
    
    Then analyze:
    1. Total revenue by product category
    2. Monthly sales trends
    3. Best performing regions
    4. Create an executive summary dashboard
    """
    
    response = llm.ask(prompt, tools=[code_tool])
    print(response.text)


if __name__ == "__main__":
    multi_tool_example()
    print("\n" + "="*60 + "\n")
    code_interpreter_with_context()
    print("\n" + "="*60 + "\n")
    practical_use_case()