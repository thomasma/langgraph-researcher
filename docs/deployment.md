# Deployment Guide

## Prerequisites

- Python 3.13+
- API Keys:
  - OpenAI API key
  - Groq API key
  - Serper API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-researcher
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

Or using uv:
```bash
uv sync
```

4. Set up environment variables:

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
SERPER_API_KEY=your-serper-api-key
```

## Running the Application

### Command Line Interface

Run the main script:
```bash
python scripts/run_research.py
```

### Demo Script

Run the demo with example topics:
```bash
python scripts/demo.py
```

### As a Module

Import and use in your Python code:

```python
import sys
sys.path.insert(0, 'src')

from scripts.run_research import run_research

result = run_research("Your research topic")
print(result["final_output"])
```

## Testing

Run security tests:
```bash
python tests/test_security.py
```

Run end-to-end injection tests:
```bash
python tests/test_injection_e2e.py
```

Run upgrade compatibility tests:
```bash
python tests/test_upgrade.py
```

## Production Deployment

### Environment Variables

Set environment variables securely:
- Use secret management services (AWS Secrets Manager, Azure Key Vault, etc.)
- Never commit `.env` files to version control
- Rotate API keys regularly

### Logging

Configure logging for production:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_system.log'),
        logging.StreamHandler()
    ]
)
```

### Error Handling

Implement comprehensive error handling:
- API rate limiting
- Network timeouts
- Invalid input handling
- LLM service failures

### Monitoring

Monitor key metrics:
- Request success/failure rates
- Processing time per agent
- API usage and costs
- Input validation failures

### Scaling Considerations

- Use async/await for concurrent requests
- Implement caching for repeated queries
- Consider batching for multiple research topics
- Use connection pooling for API clients

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

RUN pip install -e .

CMD ["python", "scripts/run_research.py"]
```

Build and run:
```bash
docker build -t langgraph-researcher .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e GROQ_API_KEY=$GROQ_API_KEY \
           -e SERPER_API_KEY=$SERPER_API_KEY \
           langgraph-researcher
```

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Import Errors
Ensure `src/` is in your Python path:
```python
import sys
sys.path.insert(0, 'src')
```

### API Key Issues
Verify environment variables are loaded:
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))  # Should not be None
```

### Rate Limiting
Implement exponential backoff:
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_llm_with_retry(llm, messages):
    return llm.invoke(messages)
```
