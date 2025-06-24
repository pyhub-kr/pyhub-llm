"""Session-based continuous analysis with Code Interpreter."""

from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter


def session_based_analysis():
    """Demonstrate session persistence for continuous analysis."""
    
    # Initialize components
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    
    # Create a session ID for this analysis
    session_id = "data_analysis_session"
    
    print("=== Step 1: Load and Prepare Data ===")
    prompt1 = """
    Create a dataset for customer analysis with:
    - 1000 customers
    - Features: age, income, spending_score, membership_years
    - Make it realistic with some correlations
    
    Store it in a DataFrame called 'customers_df'.
    Show basic info and first few rows.
    """
    
    response1 = llm.ask(
        prompt1, 
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response1.text)
    print()
    
    # Continue with same session
    print("=== Step 2: Exploratory Data Analysis ===")
    prompt2 = """
    Using the customers_df from before:
    1. Create distribution plots for all features
    2. Check for correlations
    3. Identify any outliers
    4. Save the correlation matrix as 'correlations'
    """
    
    response2 = llm.ask(
        prompt2,
        tools=[code_tool], 
        tool_kwargs={"session_id": session_id}
    )
    print(response2.text)
    print()
    
    # Continue analysis
    print("=== Step 3: Customer Segmentation ===")
    prompt3 = """
    Using the customers_df and insights from before:
    1. Perform K-means clustering (try 3-5 clusters)
    2. Find optimal number of clusters using elbow method
    3. Add cluster labels to the dataframe
    4. Visualize clusters using first 2 principal components
    5. Describe characteristics of each cluster
    """
    
    response3 = llm.ask(
        prompt3,
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response3.text)
    print()
    
    # Final step
    print("=== Step 4: Business Insights ===")
    prompt4 = """
    Based on our customer segmentation:
    1. Calculate average metrics for each cluster
    2. Create a summary table with cluster profiles
    3. Generate business recommendations for each segment
    4. Create a final visualization summarizing all findings
    """
    
    response4 = llm.ask(
        prompt4,
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response4.text)
    
    # Get session info
    print("\n=== Session Summary ===")
    session_info = code_tool.get_session_info(session_id)
    print(f"Session ID: {session_info['session_id']}")
    print(f"Number of executions: {session_info['execution_count']}")
    print(f"Files created: {session_info['created_files']}")
    
    # Cleanup session (optional)
    # code_tool.cleanup_session(session_id)


def machine_learning_workflow():
    """Example of ML workflow using sessions."""
    
    llm = OpenAILLM(model="gpt-4")
    code_tool = CodeInterpreter(backend="local")
    session_id = "ml_workflow"
    
    print("\n=== Machine Learning Workflow ===")
    
    # Step 1: Data preparation
    print("\n--- Step 1: Data Preparation ---")
    response1 = llm.ask(
        """
        Create a binary classification dataset:
        - 500 samples, 10 features
        - Some features should be informative, others noise
        - Add slight class imbalance (60-40)
        - Split into X_train, X_test, y_train, y_test (80-20)
        """,
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response1.text)
    
    # Step 2: Model training
    print("\n--- Step 2: Model Training ---")
    response2 = llm.ask(
        """
        Using the data from before:
        1. Train a Random Forest classifier
        2. Train a Logistic Regression
        3. Compare their performance
        4. Show feature importance for Random Forest
        """,
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response2.text)
    
    # Step 3: Model evaluation
    print("\n--- Step 3: Model Evaluation ---")
    response3 = llm.ask(
        """
        For the better performing model:
        1. Create confusion matrix visualization
        2. Show classification report
        3. Plot ROC curve and calculate AUC
        4. Provide interpretation of results
        """,
        tools=[code_tool],
        tool_kwargs={"session_id": session_id}
    )
    print(response3.text)


if __name__ == "__main__":
    session_based_analysis()
    print("\n" + "="*80 + "\n")
    machine_learning_workflow()