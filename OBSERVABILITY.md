# Observability Guide

This guide covers how to set up and use observability features in pyhub-llm for LLM evaluation metrics tracking, debugging, and quality monitoring.

## Overview

pyhub-llm provides a minimal, extensible observability interface that supports multiple backends:

- **LangSmith**: LangChain's evaluation and debugging platform
- **OpenTelemetry**: Industry-standard observability framework
- **Custom Providers**: Extensible architecture for adding new backends

## Quick Start

### 1. Environment-Based Auto-Configuration

The simplest way to enable observability is through environment variables:

```bash
# .env file
PYHUB_LLM_TRACE=true

# LangSmith configuration
LANGCHAIN_API_KEY=sk-your-langsmith-key
LANGCHAIN_PROJECT=my-evaluation-project

# OpenTelemetry configuration (optional)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=pyhub-llm-app
```

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.tracing import auto_configure_tracing

# Auto-configure based on environment variables
auto_configure_tracing()

# All LLM calls are now automatically traced
llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
response = llm.ask("What is the capital of France?")
```

### 2. Programmatic Configuration

For more control, configure observability programmatically:

```python
from pyhub.llm.tracing import init_tracer, LangSmithProvider

# Initialize with LangSmith
tracer = init_tracer(LangSmithProvider(
    api_key="your-api-key",
    project_name="my-project"
))

# Use with LLM
llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
```

## Installation

### Base Dependencies

The core tracing interface has no dependencies. Install providers as needed:

```bash
# For LangSmith support
pip install "pyhub-llm[langsmith]"

# For OpenTelemetry support  
pip install "pyhub-llm[tracing]"

# For both
pip install "pyhub-llm[langsmith,tracing]"
```

### Manual Dependencies

```bash
# LangSmith minimal dependencies
pip install httpx orjson

# OpenTelemetry minimal dependencies
pip install opentelemetry-api
```

## Usage Patterns

### Automatic LLM Tracing

When tracing is enabled, all LLM calls are automatically traced:

```python
from pyhub.llm import OpenAILLM, AnthropicLLM
from pyhub.llm.tracing import auto_configure_tracing

auto_configure_tracing()

# All calls to these LLMs will be traced
openai_llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
anthropic_llm = AnthropicLLM(model="claude-3-haiku-20240307", enable_tracing=True)

# These calls generate traces with:
# - Input prompt
# - Output response
# - Token usage
# - Model information
# - Timing data
response1 = openai_llm.ask("Explain quantum computing")
response2 = anthropic_llm.ask("Write a Python function")
```

### Custom Workflows with Context Managers

For complex workflows, use context managers to create custom traces:

```python
from pyhub.llm.tracing import get_tracer, SpanKind

tracer = get_tracer()

# RAG (Retrieval-Augmented Generation) pipeline
with tracer.trace("rag_pipeline", kind=SpanKind.CHAIN) as span:
    # Step 1: Retrieve documents
    with tracer.trace("document_retrieval", kind=SpanKind.RETRIEVER) as retrieval_span:
        query = "What are the benefits of renewable energy?"
        documents = vector_store.search(query, k=5)
        retrieval_span.outputs["document_count"] = len(documents)
        retrieval_span.outputs["documents"] = [doc.title for doc in documents]
    
    # Step 2: Generate context
    context = "\n\n".join(doc.content for doc in documents)
    tracer.add_event("context_prepared", context_length=len(context))
    
    # Step 3: Generate answer (automatically traced)
    prompt = f"""
    Context: {context}
    
    Question: {query}
    
    Answer based on the provided context:
    """
    
    answer = llm.ask(prompt, temperature=0.1)
    
    # Add final outputs
    span.outputs["query"] = query
    span.outputs["answer"] = answer.text
    span.metadata["source_count"] = len(documents)
```

### Function Decorators

Use decorators for automatic function tracing:

```python
from pyhub.llm.tracing import get_tracer, SpanKind

tracer = get_tracer()

@tracer.trace_function(name="document_analysis", kind=SpanKind.CHAIN)
def analyze_document(file_path: str):
    """Analyze a document using multiple LLM calls."""
    
    # Read document
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Generate summary (traced automatically)
    summary = llm.ask(f"Summarize this document:\n\n{content}")
    
    # Analyze sentiment (traced automatically) 
    sentiment = llm.ask(f"Analyze the sentiment of this text:\n\n{content}")
    
    # Extract key points (traced automatically)
    key_points = llm.ask(f"Extract 5 key points from:\n\n{content}")
    
    return {
        "summary": summary.text,
        "sentiment": sentiment.text,
        "key_points": key_points.text
    }

# Function calls are automatically traced
result = analyze_document("report.txt")
```

### Agent Workflows

For agent-like workflows with tool usage:

```python
@tracer.trace_function(name="research_agent", kind=SpanKind.AGENT)
def research_agent(topic: str):
    """Research agent that uses multiple tools."""
    
    # Plan research (traced)
    plan = llm.ask(f"Create a research plan for: {topic}")
    tracer.add_event("plan_created", plan=plan.text[:200])
    
    results = []
    
    # Execute each step
    for i, step in enumerate(plan.text.split('\n')):
        if not step.strip():
            continue
            
        with tracer.trace(f"research_step_{i+1}", kind=SpanKind.TOOL) as step_span:
            # Search for information
            search_results = web_search(step)
            step_span.outputs["search_results"] = len(search_results)
            
            # Analyze results
            analysis = llm.ask(f"Analyze these search results for '{step}':\n{search_results}")
            step_span.outputs["analysis"] = analysis.text[:500]
            
            results.append(analysis.text)
    
    # Final synthesis
    synthesis = llm.ask(f"Synthesize these research findings:\n\n" + "\n\n".join(results))
    
    return synthesis.text
```

## Multiple Providers

Use multiple observability backends simultaneously:

```python
from pyhub.llm.tracing import init_tracer, CompositeProvider
from pyhub.llm.tracing.langsmith import LangSmithProvider
from pyhub.llm.tracing.opentelemetry import OpenTelemetryProvider

# Send traces to both LangSmith and OpenTelemetry
tracer = init_tracer(CompositeProvider([
    LangSmithProvider(
        api_key="your-langsmith-key",
        project_name="evaluation-project"
    ),
    OpenTelemetryProvider(
        service_name="pyhub-llm-service"
    )
]))

# All traces now go to both systems
llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
response = llm.ask("Hello world!")
```

## Configuration Options

### LangSmith Provider Options

```python
from pyhub.llm.tracing.langsmith import LangSmithProvider

provider = LangSmithProvider(
    api_key="your-api-key",           # API key (or set LANGCHAIN_API_KEY)
    project_name="my-project",        # Project name (or set LANGCHAIN_PROJECT)  
    api_url="https://api.smith.langchain.com",  # API URL (or set LANGCHAIN_ENDPOINT)
    batch_size=20,                    # Batch size for uploads (default: 20)
    flush_interval=2.0,               # Auto-flush interval in seconds (default: 2.0)
)
```

### OpenTelemetry Provider Options

```python
from pyhub.llm.tracing.opentelemetry import OpenTelemetryProvider

provider = OpenTelemetryProvider(
    service_name="my-llm-service",    # Service name for traces
    tracer_name="pyhub.llm",         # Tracer name (default: "pyhub.llm")
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYHUB_LLM_TRACE` | Enable tracing (`true`/`false`) | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | - |
| `LANGCHAIN_PROJECT` | LangSmith project name | `default` |
| `LANGCHAIN_ENDPOINT` | LangSmith API endpoint | `https://api.smith.langchain.com` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector endpoint | - |
| `OTEL_SERVICE_NAME` | Service name for OpenTelemetry | `pyhub-llm` |

## Advanced Usage

### Custom Events and Metadata

Add custom events and metadata to traces:

```python
tracer = get_tracer()

with tracer.trace("custom_workflow", kind=SpanKind.CHAIN) as span:
    # Add custom metadata
    span.metadata["user_id"] = "user123"
    span.metadata["session_id"] = "session456"
    span.tags.extend(["production", "critical"])
    
    # Process step 1
    result1 = llm.ask("First question")
    tracer.add_event("step1_completed", 
                    result_length=len(result1.text),
                    tokens_used=result1.usage.total_tokens)
    
    # Process step 2
    result2 = llm.ask("Second question")
    tracer.add_event("step2_completed",
                    result_length=len(result2.text),
                    tokens_used=result2.usage.total_tokens)
    
    # Final outputs
    span.outputs["combined_result"] = result1.text + " " + result2.text
```

### Error Handling

Tracing automatically captures errors without breaking your application:

```python
with tracer.trace("error_handling_example") as span:
    try:
        # This might fail
        result = llm.ask("", max_tokens=0)  # Invalid parameters
    except Exception as e:
        # Error is automatically captured in the trace
        span.add_event("error_occurred", error_type=type(e).__name__)
        # Handle the error
        result = llm.ask("Fallback question")
    
    span.outputs["final_result"] = result.text
```

### Async Support

Full support for async/await patterns:

```python
import asyncio
from pyhub.llm import OpenAILLM

async def async_workflow():
    llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
    tracer = get_tracer()
    
    with tracer.trace("async_workflow", kind=SpanKind.CHAIN) as span:
        # Multiple concurrent LLM calls
        tasks = [
            llm.ask_async("Question 1"),
            llm.ask_async("Question 2"), 
            llm.ask_async("Question 3")
        ]
        
        results = await asyncio.gather(*tasks)
        span.outputs["results"] = [r.text for r in results]
    
    return results

# Run async workflow
results = asyncio.run(async_workflow())
```

## Performance Considerations

### Zero Overhead When Disabled

When tracing is disabled, there's virtually no performance impact:

```python
# No tracing overhead
llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=False)
response = llm.ask("Hello")  # Normal performance
```

### Batching and Async Uploads

Providers use batching and background uploads to minimize performance impact:

- **Batching**: Multiple traces are sent together
- **Background Tasks**: Uploads happen asynchronously
- **Error Resilience**: Trace failures don't affect main application

### Memory Management

Traces are automatically cleaned up and don't accumulate in memory:

```python
# Long-running applications won't leak memory
for i in range(10000):
    with tracer.trace(f"iteration_{i}") as span:
        result = llm.ask(f"Question {i}")
        span.outputs["result"] = result.text
    # Trace data is automatically flushed and cleaned up
```

## Troubleshooting

### Common Issues

1. **No traces appearing**
   - Check `PYHUB_LLM_TRACE=true` is set
   - Verify API keys are correct
   - Ensure `enable_tracing=True` on LLM instances

2. **Import errors**
   - Install required dependencies: `pip install "pyhub-llm[langsmith,tracing]"`
   - For minimal install, check specific provider requirements

3. **Performance issues**
   - Increase `batch_size` for providers
   - Reduce `flush_interval` if needed
   - Consider disabling tracing in production if not needed

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging

# Enable debug logging for tracing
logging.getLogger("pyhub.llm.tracing").setLevel(logging.DEBUG)

# This will show trace uploads, errors, etc.
```

### Manual Flushing

Force immediate upload of pending traces:

```python
from pyhub.llm.tracing import get_tracer

tracer = get_tracer()
tracer.provider.flush()  # Force immediate upload
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good: Descriptive names
with tracer.trace("user_query_processing", kind=SpanKind.CHAIN):
    pass

# Bad: Generic names  
with tracer.trace("process", kind=SpanKind.CHAIN):
    pass
```

### 2. Add Relevant Metadata

```python
with tracer.trace("document_summarization") as span:
    span.metadata["document_type"] = "pdf"
    span.metadata["document_pages"] = 10
    span.metadata["user_tier"] = "premium"
    span.tags.extend(["summarization", "production"])
```

### 3. Use Appropriate Span Kinds

```python
# Chain for workflows
with tracer.trace("rag_pipeline", kind=SpanKind.CHAIN):
    
    # Retriever for search/retrieval
    with tracer.trace("vector_search", kind=SpanKind.RETRIEVER):
        pass
    
    # Tool for external API calls
    with tracer.trace("web_api_call", kind=SpanKind.TOOL):
        pass
    
    # LLM calls are automatically SpanKind.LLM
```

### 4. Handle Sensitive Data

```python
with tracer.trace("user_query") as span:
    # Don't log sensitive information
    span.inputs["query_type"] = "personal_info"  # OK
    span.inputs["query"] = hash(user_query)      # Hash sensitive data
    # span.inputs["ssn"] = user_ssn              # Never do this
```

### 5. Environment-Specific Configuration

```bash
# Development
PYHUB_LLM_TRACE=true
LANGCHAIN_PROJECT=dev-experiments

# Production  
PYHUB_LLM_TRACE=true
LANGCHAIN_PROJECT=prod-monitoring
```

## Examples

### Complete RAG Application

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.tracing import auto_configure_tracing, get_tracer, SpanKind

# Setup
auto_configure_tracing()
llm = OpenAILLM(model="gpt-4o-mini", enable_tracing=True)
tracer = get_tracer()

def rag_qa_system(question: str):
    """Complete RAG Q&A system with full tracing."""
    
    with tracer.trace("rag_qa_system", kind=SpanKind.CHAIN) as main_span:
        main_span.inputs["question"] = question
        main_span.metadata["system"] = "rag_v2"
        
        # Step 1: Query preprocessing
        with tracer.trace("query_preprocessing", kind=SpanKind.TOOL) as prep_span:
            processed_query = llm.ask(f"Rephrase for search: {question}")
            prep_span.outputs["processed_query"] = processed_query.text
        
        # Step 2: Document retrieval
        with tracer.trace("document_retrieval", kind=SpanKind.RETRIEVER) as ret_span:
            documents = vector_store.search(processed_query.text, k=5)
            ret_span.outputs["document_count"] = len(documents)
            ret_span.outputs["documents"] = [doc.title for doc in documents]
        
        # Step 3: Answer generation
        context = "\n\n".join(doc.content for doc in documents)
        answer = llm.ask(f"""
        Context: {context}
        
        Question: {question}
        
        Provide a comprehensive answer based on the context:
        """)
        
        # Step 4: Answer validation
        with tracer.trace("answer_validation", kind=SpanKind.TOOL) as val_span:
            validation = llm.ask(f"""
            Question: {question}
            Answer: {answer.text}
            
            Rate this answer quality from 1-10 and explain why:
            """)
            val_span.outputs["validation"] = validation.text
        
        # Final outputs
        main_span.outputs["answer"] = answer.text
        main_span.outputs["validation_score"] = validation.text
        main_span.metadata["total_tokens"] = sum([
            processed_query.usage.total_tokens,
            answer.usage.total_tokens, 
            validation.usage.total_tokens
        ])
        
        return {
            "answer": answer.text,
            "sources": [doc.title for doc in documents],
            "validation": validation.text
        }

# Usage
result = rag_qa_system("What are the environmental benefits of solar energy?")
```

This comprehensive example shows how to trace a complete RAG system with multiple steps, proper metadata, and detailed outputs that will be valuable for evaluation and debugging.