"""
Batch processing examples.

This example demonstrates how to use the batch() method for processing
multiple prompts efficiently with different history management modes.
"""

import asyncio
import os
import time

from pyhub.llm import LLM


def ensure_api_key():
    """Ensure API key is set for the example."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)


async def independent_batch_example():
    """Example of independent batch processing (parallel execution)."""
    print("\n=== Independent Batch Processing ===")
    
    llm = LLM.create("gpt-4o-mini")
    
    prompts = [
        "What is Python? (in one sentence)",
        "What is JavaScript? (in one sentence)",
        "What is Go? (in one sentence)",
        "What is Rust? (in one sentence)",
        "What is Java? (in one sentence)"
    ]
    
    print(f"Processing {len(prompts)} prompts in parallel...")
    start_time = time.time()
    
    # Process all prompts in parallel
    replies = await llm.batch(prompts, max_parallel=3)
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds\n")
    
    for i, reply in enumerate(replies):
        print(f"Q: {prompts[i]}")
        print(f"A: {reply.text}")
        print("-" * 50)


async def sequential_batch_example():
    """Example of sequential batch processing (context builds up)."""
    print("\n=== Sequential Batch Processing ===")
    
    llm = LLM.create("gpt-4o-mini")
    
    prompts = [
        "What is the Fibonacci sequence?",
        "Write a Python function to calculate it",
        "What's the time complexity of that implementation?",
        "How can we optimize it?"
    ]
    
    print("Processing prompts sequentially (each answer provides context for next)...")
    start_time = time.time()
    
    # Process sequentially with history
    replies = await llm.batch(
        prompts,
        history_mode="sequential",
        use_history=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds\n")
    
    for i, reply in enumerate(replies):
        print(f"Step {i+1}: {prompts[i]}")
        print(f"Response: {reply.text}")
        print("-" * 50)


async def shared_context_batch_example():
    """Example of shared context batch processing."""
    print("\n=== Shared Context Batch Processing ===")
    
    llm = LLM.create("gpt-4o-mini")
    
    # Set up initial context
    print("Setting up context...")
    llm.ask(
        "I have three products: "
        "Product A is a high-performance laptop, "
        "Product B is a budget smartphone, "
        "Product C is a smart home device.",
        use_history=True
    )
    
    prompts = [
        "What are the key features of Product A?",
        "Who is the target audience for Product B?",
        "What are potential use cases for Product C?"
    ]
    
    print("\nProcessing product questions with shared context...")
    start_time = time.time()
    
    # Process with shared initial context
    replies = await llm.batch(
        prompts,
        history_mode="shared",
        use_history=True,
        max_parallel=3
    )
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds\n")
    
    for i, reply in enumerate(replies):
        print(f"Q: {prompts[i]}")
        print(f"A: {reply.text}")
        print("-" * 50)


def sync_batch_example():
    """Example of synchronous batch processing."""
    print("\n=== Synchronous Batch Processing ===")
    
    llm = LLM.create("gpt-4o-mini")
    
    prompts = [
        "What is 2 + 2?",
        "What is 10 * 5?",
        "What is 100 / 4?"
    ]
    
    print("Processing math questions synchronously...")
    start_time = time.time()
    
    # Use synchronous version
    replies = llm.batch_sync(prompts)
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds\n")
    
    for i, reply in enumerate(replies):
        print(f"{prompts[i]} = {reply.text}")


async def error_handling_example():
    """Example of batch processing with error handling."""
    print("\n=== Batch Error Handling ===")
    
    llm = LLM.create("gpt-4o-mini")
    
    # Include a prompt that might cause issues
    prompts = [
        "What is 1 + 1?",
        "Generate a response with exactly 10000 words",  # Might fail
        "What is 2 + 2?"
    ]
    
    print("Processing with error handling (fail_fast=False)...")
    
    # Continue processing even if some fail
    replies = await llm.batch(
        prompts,
        fail_fast=False,
        max_tokens=50  # Limit tokens to force potential failure
    )
    
    for i, reply in enumerate(replies):
        print(f"Q: {prompts[i]}")
        if "Error" in reply.text:
            print(f"A: [ERROR] {reply.text}")
        else:
            print(f"A: {reply.text}")
        print("-" * 50)


async def main():
    """Run all examples."""
    ensure_api_key()
    
    # Run examples
    await independent_batch_example()
    await sequential_batch_example()
    await shared_context_batch_example()
    sync_batch_example()
    await error_handling_example()
    
    print("\n✅ All batch processing examples completed!")


if __name__ == "__main__":
    asyncio.run(main())