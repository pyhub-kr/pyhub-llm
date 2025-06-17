"""Example of async image generation with pyhub-llm."""

import asyncio
from pyhub.llm import OpenAILLM, AnthropicLLM, GoogleLLM, OllamaLLM, UpstageLLM


async def demo_image_generation_async():
    """Demonstrate async image generation capabilities."""
    
    # Initialize OpenAI LLM with DALL-E 3
    llm = OpenAILLM(model="dall-e-3", api_key="your-api-key")
    
    # Test if model supports image generation
    if llm.supports("image_generation"):
        print(f"Model {llm.model} supports image generation")
        print(f"Supported sizes: {llm.get_supported_image_sizes()}")
        
        try:
            # Generate image asynchronously
            print("\nGenerating image asynchronously...")
            reply = await llm.generate_image_async(
                "A futuristic cityscape at sunset with flying cars",
                size="1024x1792",
                quality="hd",
                style="vivid"
            )
            
            print(f"Image URL: {reply.url}")
            print(f"Revised prompt: {reply.revised_prompt}")
            print(f"Size: {reply.size} ({reply.width}x{reply.height})")
            
        except Exception as e:
            print(f"Error generating image: {e}")
    
    # Test providers that don't support image generation
    unsupported_providers = [
        ("Anthropic", AnthropicLLM(model="claude-3-opus", api_key="test-key")),
        ("Google", GoogleLLM(model="gemini-pro", api_key="test-key")),
        ("Ollama", OllamaLLM(model="llama2")),
        ("Upstage", UpstageLLM(model="solar-mini", api_key="test-key"))
    ]
    
    print("\n\nTesting unsupported providers:")
    for name, llm in unsupported_providers:
        print(f"\n{name}:")
        print(f"  Supports image generation: {llm.supports('image_generation')}")
        try:
            await llm.generate_image_async("test")
        except NotImplementedError as e:
            print(f"  Expected error: {e}")


async def demo_concurrent_operations():
    """Demonstrate concurrent async operations."""
    llm = OpenAILLM(model="dall-e-3", api_key="your-api-key")
    
    # Create multiple image generation tasks
    prompts = [
        "A serene mountain landscape",
        "An abstract painting with vibrant colors",
        "A steampunk mechanical dragon"
    ]
    
    print("Generating multiple images concurrently...")
    
    # Run all tasks concurrently
    tasks = [llm.generate_image_async(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Image {i+1} failed: {result}")
        else:
            print(f"Image {i+1} generated: {result.url}")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(demo_image_generation_async())
    
    # Uncomment to run concurrent operations demo
    # asyncio.run(demo_concurrent_operations())