#!/usr/bin/env python
"""
Test script to verify streaming Reply objects functionality.
Tests both with and without Rich library for markdown rendering.
"""

import os
import sys
from typing import Generator

# Add src to path for development testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pyhub.llm import LLM, display
from pyhub.llm.types import Reply

# Check if API keys are available
api_keys = {
    "OPENAI_API_KEY": "gpt-4o-mini",
    "ANTHROPIC_API_KEY": "claude-3-haiku-20240307",
}

available_models = []
for env_var, model in api_keys.items():
    if os.getenv(env_var):
        available_models.append(model)
    else:
        print(f"⚠️  {env_var} not set. Skipping {model} tests.")

if not available_models:
    print("❌ No API keys found. Please set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY")
    sys.exit(1)

print(f"✅ Found API keys for: {', '.join(available_models)}")
print()

# Check Rich availability
try:
    import rich
    HAS_RICH = True
    print("✅ Rich library is installed")
except ImportError:
    HAS_RICH = False
    print("⚠️  Rich library not installed. Testing plain text mode only.")
    print("   Install with: pip install rich")
print()


def test_streaming_type_handling():
    """Test that streaming responses yield Reply objects, not strings."""
    print("=" * 60)
    print("1. Testing Streaming Type Handling")
    print("=" * 60)
    
    for model_name in available_models:
        print(f"\n📌 Testing with {model_name}:")
        llm = LLM.create(model_name)
        
        # Get streaming response
        response = llm.ask("Count from 1 to 3 slowly", stream=True)
        
        # Check the type of yielded chunks
        print("   Checking chunk types...")
        chunk_count = 0
        has_reply_objects = True
        
        for chunk in response:
            chunk_count += 1
            if isinstance(chunk, Reply):
                print(f"   ✅ Chunk {chunk_count}: Reply object with text='{chunk.text}'")
            elif isinstance(chunk, str):
                print(f"   ⚠️  Chunk {chunk_count}: String '{chunk}' (should be Reply)")
                has_reply_objects = False
            else:
                print(f"   ❌ Chunk {chunk_count}: Unknown type {type(chunk)}")
                has_reply_objects = False
        
        if has_reply_objects:
            print(f"   ✅ All chunks are Reply objects")
        else:
            print(f"   ❌ Not all chunks are Reply objects")


def test_display_with_streaming():
    """Test display() function with streaming responses."""
    print("\n" + "=" * 60)
    print("2. Testing display() with Streaming")
    print("=" * 60)
    
    for model_name in available_models:
        print(f"\n📌 Testing with {model_name}:")
        llm = LLM.create(model_name)
        
        # Test plain text display
        print("\n   Plain text streaming:")
        print("   ", end="")
        response = llm.ask("Write 'Hello World' in 3 languages", stream=True)
        text = display(response, markdown=False)
        print(f"\n   ✅ Collected text length: {len(text)} chars")
        
        # Test markdown display if Rich is available
        if HAS_RICH:
            print("\n   Markdown streaming:")
            response = llm.ask("Write 'Hello' in **bold** and `code`", stream=True)
            text = display(response, markdown=True)
            print(f"\n   ✅ Collected text length: {len(text)} chars")


def test_reply_print_method():
    """Test Reply.print() method for non-streaming responses."""
    print("\n" + "=" * 60)
    print("3. Testing Reply.print() Method")
    print("=" * 60)
    
    for model_name in available_models:
        print(f"\n📌 Testing with {model_name}:")
        llm = LLM.create(model_name)
        
        # Get non-streaming response
        reply = llm.ask("Say 'test passed' in markdown with **bold**")
        
        # Test plain text print
        print("\n   Plain text print:")
        reply.print(markdown=False)
        
        # Test markdown print if Rich is available
        if HAS_RICH:
            print("\n   Markdown print:")
            reply.print(markdown=True)
        
        print(f"\n   ✅ Reply text length: {len(reply.text)} chars")


def test_streaming_with_usage():
    """Test that streaming responses handle usage information correctly."""
    print("\n" + "=" * 60)
    print("4. Testing Streaming with Usage Information")
    print("=" * 60)
    
    for model_name in available_models:
        print(f"\n📌 Testing with {model_name}:")
        llm = LLM.create(model_name)
        
        response = llm.ask("Count to 5", stream=True)
        
        chunks = list(response)
        print(f"   Total chunks: {len(chunks)}")
        
        # Check if last chunk has usage info
        if chunks:
            last_chunk = chunks[-1]
            if hasattr(last_chunk, 'usage') and last_chunk.usage:
                print(f"   ✅ Usage info found: input={last_chunk.usage.input}, output={last_chunk.usage.output}")
            else:
                print(f"   ⚠️  No usage info in last chunk")
        
        # Display the full text
        full_text = "".join(chunk.text for chunk in chunks if hasattr(chunk, 'text'))
        print(f"   Full text: '{full_text}'")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("5. Testing Edge Cases")
    print("=" * 60)
    
    # Test empty response handling
    print("\n📌 Testing empty/minimal responses:")
    
    for model_name in available_models[:1]:  # Just test with first available model
        llm = LLM.create(model_name)
        
        # Very short response
        response = llm.ask("Reply with just 'OK'", stream=True)
        chunks = list(response)
        print(f"   Short response chunks: {len(chunks)}")
        
        # Test display with empty generator
        def empty_generator():
            return
            yield  # Make it a generator
        
        try:
            text = display(empty_generator())
            print(f"   ✅ Empty generator handled: returned '{text}'")
        except Exception as e:
            print(f"   ❌ Empty generator error: {e}")


def test_manual_streaming():
    """Test manual iteration over streaming responses."""
    print("\n" + "=" * 60)
    print("6. Testing Manual Streaming Iteration")
    print("=" * 60)
    
    for model_name in available_models[:1]:  # Just test with first available model
        print(f"\n📌 Testing with {model_name}:")
        llm = LLM.create(model_name)
        
        response = llm.ask("Count 1, 2, 3", stream=True)
        
        # Manual iteration
        print("   Manual iteration:")
        collected_text = ""
        for i, chunk in enumerate(response):
            if hasattr(chunk, 'text'):
                collected_text += chunk.text
                print(f"     Chunk {i}: '{chunk.text}' (Reply object)")
            else:
                collected_text += str(chunk)
                print(f"     Chunk {i}: '{chunk}' (type: {type(chunk).__name__})")
        
        print(f"   ✅ Total collected: '{collected_text}'")


def main():
    """Run all tests."""
    print("🧪 Testing Streaming Reply Objects Fix")
    print("=" * 60)
    print()
    
    # Run all tests
    test_streaming_type_handling()
    test_display_with_streaming()
    test_reply_print_method()
    test_streaming_with_usage()
    test_edge_cases()
    test_manual_streaming()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\nSummary:")
    print("- Streaming responses should yield Reply objects, not strings")
    print("- display() function handles both Reply objects and strings")
    print("- Rich library enables markdown rendering when available")
    print("- Usage information may be in the last chunk (provider-dependent)")


if __name__ == "__main__":
    main()