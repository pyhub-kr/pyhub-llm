"""
Display functionality demonstration for pyhub-llm.

This example shows how to use the new display() function and print() methods
for better output formatting with optional markdown rendering.
"""
import os
from pyhub.llm import LLM, display

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def basic_display_examples():
    """Basic display functionality examples."""
    print("=== Basic Display Examples ===\n")
    
    llm = LLM.create("gpt-4o-mini")
    
    # Example 1: Regular response with markdown
    print("1. Regular response with markdown rendering:")
    reply = llm.ask("Write a short markdown example with # heading, **bold**, and `code`")
    reply.print()  # Default: markdown=True
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Same response without markdown
    print("2. Same response as plain text:")
    reply.print(markdown=False)
    
    print("\n" + "-"*50 + "\n")


def streaming_display_examples():
    """Streaming display examples."""
    print("=== Streaming Display Examples ===\n")
    
    llm = LLM.create("gpt-4o-mini")
    
    # Example 3: Streaming with markdown
    print("3. Streaming with markdown rendering:")
    response = llm.ask(
        "Write a Python function to calculate factorial with markdown formatting",
        stream=True
    )
    text = display(response)  # Automatically renders markdown while streaming
    
    print("\n" + "-"*50 + "\n")
    
    # Example 4: Streaming with plain text
    print("4. Streaming with plain text:")
    response = llm.ask(
        "Explain recursion in 2 sentences",
        stream=True
    )
    text = display(response, markdown=False)
    
    print("\n" + "-"*50 + "\n")


def advanced_display_examples():
    """Advanced display options."""
    print("=== Advanced Display Examples ===\n")
    
    llm = LLM.create("gpt-4o-mini")
    
    # Example 5: Code with syntax highlighting
    print("5. Code example with syntax highlighting:")
    response = llm.ask("""
    Write a Python class example for a simple Calculator with add and multiply methods.
    Use proper markdown code blocks.
    """)
    response.print()
    
    print("\n" + "-"*50 + "\n")
    
    # Example 6: Table formatting
    print("6. Markdown table example:")
    response = llm.ask("""
    Create a markdown table comparing Python, JavaScript, and Go with columns:
    Language, Type System, Main Use Case
    """)
    response.print()
    
    print("\n" + "-"*50 + "\n")


def custom_console_example():
    """Example with custom Rich console settings."""
    print("=== Custom Console Example ===\n")
    
    try:
        from rich.console import Console
        
        # Create custom console with specific width
        console = Console(width=80)
        
        llm = LLM.create("gpt-4o-mini")
        
        print("7. Custom console width (80 chars):")
        response = llm.ask("Write a long paragraph about AI")
        display(response, console=console)
        
    except ImportError:
        print("Rich library not installed. Skipping custom console example.")
        print("Install with: pip install rich")
    
    print("\n" + "-"*50 + "\n")


def error_handling_example():
    """Example showing fallback when Rich is not available."""
    print("=== Error Handling Example ===\n")
    
    # This will work even without Rich installed
    llm = LLM.create("gpt-4o-mini")
    
    print("8. Works even without Rich library:")
    response = llm.ask("Say hello")
    response.print()  # Falls back to plain text if Rich not available
    
    print("\n" + "="*50 + "\n")


def main():
    """Run all examples."""
    print("🎨 pyhub-llm Display Feature Demo\n")
    print("This demo shows the new display() function and print() methods.\n")
    
    # Run examples
    basic_display_examples()
    streaming_display_examples()
    advanced_display_examples()
    custom_console_example()
    error_handling_example()
    
    print("\n✅ Demo completed!")
    print("\nKey takeaways:")
    print("1. Use response.print() for simple markdown rendering")
    print("2. Use display(response) for more control")
    print("3. Both work with streaming and non-streaming responses")
    print("4. Markdown rendering requires 'rich' library (pip install rich)")
    print("5. Falls back to plain text if Rich is not available")


if __name__ == "__main__":
    main()