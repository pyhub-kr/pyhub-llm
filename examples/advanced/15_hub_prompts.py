"""
Hub functionality examples for pyhub-llm.

This example demonstrates:
1. Pulling prompts from the hub
2. Using prompts with LLMs
3. Creating and saving custom prompts
"""

from pyhub.llm import hub, LLM
from pyhub.llm.templates import PromptTemplate

def example_rag_prompt():
    """Example using the RAG prompt template."""
    print("=== RAG Prompt Example ===")
    
    # Pull the RAG prompt from hub
    rag_prompt = hub.pull("rlm/rag-prompt")
    print(f"Prompt variables: {rag_prompt.input_variables}")
    print(f"Metadata: {rag_prompt.metadata}")
    
    # Format the prompt with context and question
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.
    The tower is 330 meters (1,083 ft) tall and is the tallest structure in Paris.
    """
    
    question = "How tall is the Eiffel Tower?"
    
    formatted_prompt = rag_prompt.format(context=context, question=question)
    print(f"\nFormatted prompt:\n{formatted_prompt}")
    
    # Use with LLM
    llm = LLM.create("gpt-4o-mini")
    response = llm.ask(formatted_prompt)
    print(f"\nLLM Response: {response}")


def example_react_agent():
    """Example using the ReAct agent prompt."""
    print("\n=== ReAct Agent Prompt Example ===")
    
    # Pull the ReAct prompt
    react_prompt = hub.pull("hwchase17/react")
    print(f"Prompt variables: {react_prompt.input_variables}")
    
    # Format with tools information
    tools = """
    Calculator: Useful for performing mathematical calculations.
    Search: Useful for searching information on the internet.
    """
    
    tool_names = "Calculator, Search"
    input_question = "What is the square root of the year the Eiffel Tower was completed?"
    
    formatted_prompt = react_prompt.format(
        tools=tools,
        tool_names=tool_names,
        input=input_question
    )
    print(f"\nFormatted prompt (first 500 chars):\n{formatted_prompt[:500]}...")


def example_custom_prompt():
    """Example creating and saving a custom prompt."""
    print("\n=== Custom Prompt Example ===")
    
    # Create a custom prompt
    custom_prompt = PromptTemplate(
        template="""You are an expert {role} assistant.
        
Task: {task}

Requirements:
{requirements}

Please provide a detailed response:""",
        input_variables=["role", "task", "requirements"],
        metadata={
            "description": "Expert assistant prompt",
            "author": "pyhub-llm",
            "tags": ["expert", "assistant"],
            "version": "1.0.0"
        }
    )
    
    # Save to hub
    hub.push("custom/expert-assistant", custom_prompt)
    print("Custom prompt saved to hub!")
    
    # Pull it back
    loaded_prompt = hub.pull("custom/expert-assistant")
    
    # Use the prompt
    formatted = loaded_prompt.format(
        role="Python developer",
        task="Review the following code for best practices",
        requirements="Focus on readability, performance, and security"
    )
    print(f"\nFormatted custom prompt:\n{formatted}")


def example_list_prompts():
    """Example listing all available prompts."""
    print("\n=== Available Prompts ===")
    
    prompts = hub.list_prompts()
    for prompt_name in prompts:
        print(f"- {prompt_name}")


def example_partial_prompt():
    """Example using partial prompts."""
    print("\n=== Partial Prompt Example ===")
    
    # Pull a prompt and create a partial version
    rag_prompt = hub.pull("rlm/rag-prompt")
    
    # Pre-fill the context
    context = """
    Python is a high-level, interpreted programming language.
    It was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability and simplicity.
    """
    
    # Create a partial prompt with context pre-filled
    python_qa_prompt = rag_prompt.partial(context=context)
    print(f"Original variables: {rag_prompt.input_variables}")
    print(f"Partial variables: {python_qa_prompt.input_variables}")
    
    # Now we only need to provide the question
    questions = [
        "Who created Python?",
        "When was Python released?",
        "What does Python emphasize?"
    ]
    
    llm = LLM.create("gpt-4o-mini")
    for question in questions:
        formatted = python_qa_prompt.format(question=question)
        response = llm.ask(formatted)
        print(f"\nQ: {question}")
        print(f"A: {response}")


if __name__ == "__main__":
    # Run examples
    example_rag_prompt()
    example_react_agent()
    example_custom_prompt()
    example_list_prompts()
    example_partial_prompt()