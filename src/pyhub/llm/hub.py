"""Hub functionality for pulling and managing prompts."""

import json
import os
from pathlib import Path
from typing import Optional, Union, Dict, List

from .templates import PromptTemplate


class PromptHub:
    """Hub for managing and pulling prompts."""
    
    # Built-in popular prompts
    BUILTIN_PROMPTS = {
        "rlm/rag-prompt": {
            "template": """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:""",
            "input_variables": ["context", "question"],
            "template_format": "f-string",
            "metadata": {
                "description": "RAG prompt for question answering with context",
                "author": "rlm",
                "tags": ["rag", "question-answering"],
                "version": "1.0.0"
            }
        },
        "hwchase17/react": {
            "template": """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:""",
            "input_variables": ["tools", "tool_names", "input"],
            "template_format": "f-string",
            "metadata": {
                "description": "ReAct agent prompt for reasoning and acting",
                "author": "hwchase17",
                "tags": ["agent", "react", "tools"],
                "version": "1.0.0"
            }
        },
        "hwchase17/openai-functions-agent": {
            "template": """You are a helpful AI assistant. You have access to the following tools:

{tools}

You must always use one of the provided tools to answer questions. The tools are provided in OpenAI function format.

Current conversation:
{history}

User: {input}
Assistant:""",
            "input_variables": ["tools", "history", "input"],
            "template_format": "f-string",
            "metadata": {
                "description": "OpenAI functions agent prompt",
                "author": "hwchase17",
                "tags": ["agent", "openai-functions", "tools"],
                "version": "1.0.0"
            }
        },
        "hwchase17/structured-chat-agent": {
            "template": """System: Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation

{history}
Human: {input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)""",
            "input_variables": ["tools", "tool_names", "history", "input", "agent_scratchpad"],
            "template_format": "f-string",
            "metadata": {
                "description": "Structured chat agent with JSON output",
                "author": "hwchase17",
                "tags": ["agent", "structured-output", "json", "tools"],
                "version": "1.0.0"
            }
        }
    }
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize PromptHub.
        
        Args:
            cache_dir: Directory to cache prompts locally
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default cache directory
            self.cache_dir = Path.home() / ".pyhub" / "prompts"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def pull(self, prompt_name: str, version: Optional[str] = None) -> PromptTemplate:
        """Pull a prompt from the hub.
        
        Args:
            prompt_name: Name of the prompt (e.g., "rlm/rag-prompt")
            version: Optional version to pull (default: latest)
            
        Returns:
            PromptTemplate instance
            
        Raises:
            ValueError: If prompt not found
        """
        # Check built-in prompts first
        if prompt_name in self.BUILTIN_PROMPTS:
            prompt_data = self.BUILTIN_PROMPTS[prompt_name]
            return PromptTemplate(
                template=prompt_data["template"],
                input_variables=prompt_data["input_variables"],
                template_format=prompt_data.get("template_format", "f-string"),
                metadata=prompt_data.get("metadata", {})
            )
        
        # Check local cache
        cache_file = self.cache_dir / f"{prompt_name.replace('/', '_')}.json"
        if cache_file.exists():
            return PromptTemplate.load(cache_file)
        
        # TODO: In the future, could fetch from remote hub
        raise ValueError(f"Prompt '{prompt_name}' not found in hub")
    
    def push(self, prompt_name: str, prompt: PromptTemplate) -> None:
        """Push a prompt to local cache.
        
        Args:
            prompt_name: Name for the prompt
            prompt: PromptTemplate to save
        """
        # Save to local cache
        cache_file = self.cache_dir / f"{prompt_name.replace('/', '_')}.json"
        prompt.save(cache_file)
    
    def list_prompts(self) -> List[str]:
        """List all available prompts.
        
        Returns:
            List of prompt names
        """
        prompts = list(self.BUILTIN_PROMPTS.keys())
        
        # Add locally cached prompts
        for cache_file in self.cache_dir.glob("*.json"):
            prompt_name = cache_file.stem.replace("_", "/")
            if prompt_name not in prompts:
                prompts.append(prompt_name)
        
        return sorted(prompts)


# Global hub instance
hub = PromptHub()


# Convenience functions
def pull(prompt_name: str, version: Optional[str] = None) -> PromptTemplate:
    """Pull a prompt from the hub.
    
    Args:
        prompt_name: Name of the prompt (e.g., "rlm/rag-prompt")
        version: Optional version to pull (default: latest)
        
    Returns:
        PromptTemplate instance
    """
    return hub.pull(prompt_name, version)


def push(prompt_name: str, prompt: PromptTemplate) -> None:
    """Push a prompt to local cache.
    
    Args:
        prompt_name: Name for the prompt
        prompt: PromptTemplate to save
    """
    hub.push(prompt_name, prompt)


def list_prompts() -> List[str]:
    """List all available prompts.
    
    Returns:
        List of prompt names
    """
    return hub.list_prompts()