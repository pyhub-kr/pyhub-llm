"""Docker backend example for Code Interpreter."""

from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter


def docker_backend_example():
    """Demonstrate using Docker backend for isolated execution."""
    
    # Initialize components
    llm = OpenAILLM(model="gpt-4")
    
    # Create Code Interpreter with Docker backend
    # Note: Docker must be installed and running
    code_tool = CodeInterpreter(
        backend="docker",
        backend_config={
            "image_name": "python:3.9-slim",
            "memory_limit": "512m",
            "cpu_quota": 50000,  # 50% of one CPU
            "network_mode": "none",  # No network access for security
        }
    )
    
    print("=== Docker Backend Example ===")
    print("Note: This requires Docker to be installed and running\n")
    
    # Example 1: Basic execution in Docker
    print("--- Example 1: Isolated Execution ---")
    prompt1 = """
    Show that we're running in an isolated Docker container:
    1. Print Python version
    2. Show available packages
    3. Create a simple plot to test matplotlib
    4. Try to access the network (should fail)
    """
    
    try:
        response1 = llm.ask(prompt1, tools=[code_tool])
        print(response1.text)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Docker is installed and running")
    
    print("\n--- Example 2: Complex Analysis in Docker ---")
    prompt2 = """
    Perform a Monte Carlo simulation:
    1. Simulate 10,000 random walks
    2. Calculate statistics
    3. Create visualization
    4. Show that the container is isolated from host system
    """
    
    try:
        response2 = llm.ask(prompt2, tools=[code_tool])
        print(response2.text)
    except Exception as e:
        print(f"Error: {e}")


def remote_docker_example():
    """Example of using remote Docker daemon."""
    
    llm = OpenAILLM(model="gpt-4")
    
    # Connect to remote Docker daemon
    # Replace with your remote Docker URL
    remote_docker_url = "tcp://remote-docker-host:2376"
    
    print("\n=== Remote Docker Backend Example ===")
    print(f"Connecting to remote Docker at: {remote_docker_url}\n")
    
    try:
        code_tool = CodeInterpreter(
            backend="docker",
            backend_config={
                "remote_docker_url": remote_docker_url,
                "image_name": "python:3.9-slim",
                "memory_limit": "1g",  # More resources on remote server
            }
        )
        
        prompt = """
        Run a resource-intensive computation:
        1. Generate large dataset (1M points)
        2. Perform statistical analysis
        3. Create visualizations
        """
        
        response = llm.ask(prompt, tools=[code_tool])
        print(response.text)
        
    except Exception as e:
        print(f"Error connecting to remote Docker: {e}")
        print("Make sure the remote Docker daemon is accessible")


def docker_vs_local_comparison():
    """Compare Docker and Local backends."""
    
    llm = OpenAILLM(model="gpt-4")
    
    print("\n=== Docker vs Local Backend Comparison ===")
    
    # Test code that tries to access system resources
    test_code = """
    Show system information and try various operations:
    1. Import os and sys
    2. Try to list files in home directory
    3. Try to make network request
    4. Show current working directory
    5. Check available memory
    """
    
    # Local backend
    print("\n--- Local Backend (Restricted) ---")
    local_tool = CodeInterpreter(backend="local")
    response_local = llm.ask(test_code, tools=[local_tool])
    print(response_local.text)
    
    # Docker backend
    print("\n--- Docker Backend (Isolated) ---")
    try:
        docker_tool = CodeInterpreter(backend="docker")
        response_docker = llm.ask(test_code, tools=[docker_tool])
        print(response_docker.text)
    except Exception as e:
        print(f"Docker error: {e}")


if __name__ == "__main__":
    # Check if Docker is available
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("Docker is available!\n")
        
        docker_backend_example()
        print("\n" + "="*60 + "\n")
        docker_vs_local_comparison()
        
        # Uncomment to test remote Docker
        # print("\n" + "="*60 + "\n")
        # remote_docker_example()
        
    except ImportError:
        print("Docker package not installed.")
        print("Install with: pip install pyhub-llm[docker]")
    except Exception as e:
        print(f"Docker not available: {e}")
        print("Make sure Docker Desktop is installed and running")