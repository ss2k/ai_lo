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
- Python 3.11+
- OpenAI API key
- (Optional) LangSmith API key for tracing

### Setup

1. **Clone the repository**
```bash
cd ai_loan_officer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Copy sample.env file and change it to `.env` and put in the correct credentials.

4. **Load company documents into vector database**
```bash
python load_data.py
```

This will:
- Load all `.md` files from `docs/` directory
- Split them into chunks (512 chars with 200 overlap)
- Create embeddings using OpenAI
- Store in Chroma vector database at `chroma_db/`
- Needs to be done only the first time.

## Usage

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
