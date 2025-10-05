# 🏦 AI Loan Officer

An intelligent mortgage loan assistant powered by LangChain, LangGraph, and RAG that can answer questions about loan products and guide users through the mortgage application process.

## Features

### 💬 Conversational Q&A
- Answer questions about loan products using company documentation
- Retrieval-Augmented Generation (RAG) with Chroma vector database
- Automatic fallback to general knowledge when company docs don't have the answer
- Topic validation to keep conversations on track

### 📝 Mortgage Application Flow
- Interactive step-by-step mortgage application process
- Collects: credit score, home value, down payment, income, debts, loan term
- Subprime handling (credit score < 620) with user confirmation
- Supports both dollar amounts and percentages for down payment

### 🔢 Rate Calculation
- Automated interest rate calculation based on:
  - Credit score
  - Loan-to-Value (LTV) ratio
  - Debt-to-Income (DTI) ratio
  - Loan term (15 or 30 years)
- LangChain Tool integration for modular rate calculation

## Tech Stack

- **LLMs**: OpenAI GPT-4 Turbo
- **Orchestration**: LangGraph for stateful workflows
- **RAG**: LangChain + Chroma vector database
- **Embeddings**: OpenAI text-embedding-3-small
- **Observability**: LangSmith tracing
- **UI**: Streamlit
- **Data**: Pandas for rate matrix handling

## Architecture

### Workflow Structure
```
User Input → Topic Validation → Intent Routing
                                     ↓
                        ┌────────────┴────────────┐
                        ↓                         ↓
                    Q&A Flow              Application Flow
                        ↓                         ↓
            RAG → Relevance Check      Step-by-step Data Collection
                        ↓                         ↓
                  LLM Answer              Rate Calculation Tool
```

### Key Components

- **`workflow.py`**: LangGraph workflow with routing logic
- **`retriever.py`**: RAG document retrieval from Chroma
- **`rate_calculator.py`**: Rate matching service
- **`rate_tool.py`**: LangChain Tool wrapper for rate calculation
- **`load_data.py`**: Script to embed documents into Chroma
- **`app.py`**: Streamlit web interface
- **`main.py`**: CLI interface (currently commented out)

## Installation

### Prerequisites
- Python 3.11+ OR Docker
- OpenAI API key
- (Optional) LangSmith API key for tracing

### Option 1: Docker (Recommended)

1. **Clone the repository**
```bash
cd ai_lo
```

2. **Using sample .env, set up your API keys in the environment variables**
```bash
cp sample.env .env
# Edit .env and add your API keys
```

3. **Load company documents into vector database**
```bash
# First time only - run this outside Docker
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python load_data.py
deactivate
```

4. **Build and run with Docker**
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

Navigate to `http://localhost:8501`

**Note**: The `chroma_db/` directory is mounted as a volume, so your vector database persists across container restarts.

### Option 2: Local Python Installation

1. **Clone the repository**
```bash
cd ai_lo
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Using sample .env, set up your API keys in the environment variables**
```bash
cp sample.env .env
# Edit .env and add your API keys
```

5. **Load company documents into vector database**
```bash
python load_data.py
```

This will:
- Load all `.md` files from `docs/` directory
- Split them into chunks (512 chars with 200 overlap)
- Create embeddings using OpenAI
- Store in Chroma vector database at `chroma_db/`
- Needs to be done only the first time.

6. **Run the application**
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

## Configuration

### Rate Matrix
Edit `docs/rate_matrix.csv` to update rate criteria:
```csv
min_credit,max_ltv,max_dti,loan_term,rate
760,60,36,30,6.500
...
```

### Document Sources
Add `.md` files to `docs/` directory and re-run `load_data.py`

### Model Configuration
Change LLM in `workflow.py`:
```python
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
```

## Testing

Run the test suite:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_rate_calculator.py

# With verbose output
pytest -v
```

## Docker Commands

### Useful Docker Commands

```bash
# Build the image
docker-compose build

# Start containers in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Access container shell
docker-compose exec ai-loan-officer bash

# View running containers
docker ps

# Remove all stopped containers and unused images
docker system prune -a
```

### Troubleshooting

**Container won't start:**
```bash
# Check logs
docker-compose logs ai-loan-officer

# Check if port 8501 is already in use
lsof -i :8501  # On macOS/Linux
netstat -ano | findstr :8501  # On Windows
```

**Chroma DB issues:**
```bash
# Ensure chroma_db directory exists and has correct permissions
chmod -R 755 chroma_db

# Recreate the vector database
rm -rf chroma_db/*
python load_data.py
```

**Environment variables not loading:**
```bash
# Verify .env file exists
ls -la .env

# Check if variables are set in container
docker-compose exec ai-loan-officer env | grep OPENAI
```
