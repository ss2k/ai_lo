from typing import TypedDict, List, Optional, Literal
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriever import get_retriever
from tools.rate_tool import rate_calculation_tool
import os
import re

if os.getenv("LANGCHAIN_TRACING_V2"):
    print("[INFO] LangSmith tracing is enabled")

class AgentState(TypedDict):
    """State structure for the loan officer agent workflow."""

    # Conversation data
    messages: List[BaseMessage]
    user_input: str

    # Mode/routing
    mode: Literal["qa", "application", "error"]
    intent: Optional[str]

    # Retrieved context
    retrieved_docs: Optional[List]
    context: Optional[str]

    # Application data
    credit_score: Optional[int]
    home_value: Optional[int]
    down_payment: Optional[int]
    loan_amount: Optional[int]
    income: Optional[int]
    debts: Optional[float]
    loan_term: Optional[int]
    application_step: Optional[str]
    subprime_continue: Optional[bool]

    # Results
    final_response: Optional[str]
    calculated_rate: Optional[float]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Helper function
def extract_number(text: str, field_name: str) -> Optional[float]:
    """Extract a number from text using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a data extraction assistant. Extract the {field_name} from the user's message.

        IMPORTANT: Return ONLY the numeric value as digits. No words, no explanations, no units.
        If you cannot find a {field_name}, respond with 'NONE'.

        Examples:
        - "I want $500,000" -> 500000
        - "My credit score is 750" -> 750
        - "about 100000" -> 100000
        - "1000" -> 1000
        - "Its about 1000" -> 1000
        - "$1,000" -> 1000
        - "I don't know" -> NONE"""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({"input": text})

    response = result.content.strip()

    if response.upper() == "NONE":
        return None

    # Clean response: remove any non-digit characters except decimal point
    cleaned = ''.join(c for c in response if c.isdigit() or c == '.')

    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

# Node functions
def validate_topic(state: AgentState) -> AgentState:
    """Check if the question is related to loans/mortgages."""
    user_input_lower = state["user_input"].lower()

    application_keywords = ["application", "apply", "mortgage", "loan", "rate", "borrow", "finance", "refinance"]
    if any(keyword in user_input_lower for keyword in application_keywords):
        # If mode was in error state, we need to reset it, otherwise it will get stuck in a loop
        if state.get("mode") == "error":
            state["mode"] = "qa"
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a topic validator for a mortgage/loan officer assistant. Determine if the user's message is related to loans, mortgages, or home financing.

        Respond with only 'yes' if it's related to loans/mortgages, or 'no' if it's about something completely unrelated (like weather, sports, etc.)."""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})

    is_relevant = result.content.strip().lower() == "yes"

    if not is_relevant:
        state["final_response"] = "Sorry, I am only an expert in loan and mortgage related things. Please ask a question related to that or let me know if you want to start an application."
        state["mode"] = "error"
    else:
        # Reset mode if it was previously in error state
        if state.get("mode") == "error":
            state["mode"] = "qa"

    return state

def route_intent(state: AgentState) -> AgentState:
    """Determine if user wants Q&A or to start application."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant. Determine if the user wants to:
        1. Ask a question about loans (respond with 'qa')
        2. Start a mortgage application process (respond with 'application')

        User wants to START AN APPLICATION if they say things like:
        - "I want a loan"
        - "I want to apply"
        - "I need a mortgage"
        - "I want to start an application"
        - "Can I get a loan?"
        - "I'd like to borrow money"

        User wants Q&A if they ask informational questions like:
        - "What are your rates?"
        - "How does the process work?"
        - "What documents do I need?"
        - "Tell me about your loan terms"

        Respond with only 'qa' or 'application'."""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})

    mode = result.content.strip().lower()
    state["mode"] = mode if mode in ["qa", "application"] else "qa"

    return state

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve relevant documents from vector store."""
    retriever = get_retriever(k=3)
    docs = retriever.invoke(state["user_input"])

    state["retrieved_docs"] = docs
    state["context"] = "\n\n".join([doc.page_content for doc in docs])

    return state

def check_relevance(state: AgentState) -> AgentState:
    """Check if retrieved documents are relevant to the question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance evaluator. Determine if the provided context contains information that can answer the user's question.

        Context:
        {context}

        Question: {question}

        Respond with only 'yes' if the context contains relevant information to answer the question, or 'no' if it does not."""),
        ("human", "Is the context relevant?")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "context": state.get("context", ""),
        "question": state["user_input"]
    })

    is_relevant = result.content.strip().lower() == "yes"

    if not is_relevant:
        state["context"] = None

    return state

def answer_question(state: AgentState) -> AgentState:
    """Generate answer using retrieved documents or general knowledge."""
    context = state.get("context")
    using_internal_docs = context is not None

    # if we find the relevant answer in the internal docs, we use that, otherwise we fall back to general knowledge
    if using_internal_docs:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful loan officer assistant. Answer the user's question based on the provided context from company documents.

            Context:
            {context}"""),
            ("human", "{question}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful loan officer assistant. Answer the user's question using your general knowledge about loans and mortgages.

            IMPORTANT: Mention that this information might not reflect company-specific policies and the user should verify with official documentation."""),
            ("human", "{question}")
        ])

    chain = prompt | llm

    if using_internal_docs:
        result = chain.invoke({
            "context": context,
            "question": state["user_input"]
        })
    else:
        result = chain.invoke({
            "question": state["user_input"]
        })

    state["final_response"] = result.content

    return state

def start_application(state: AgentState) -> AgentState:
    """Start the mortgage application process."""
    state["final_response"] = "Great! Let's start your mortgage application. First, what is your credit score?"
    state["application_step"] = "credit_score"

    return state

def process_credit_score(state: AgentState) -> AgentState:
    """Process credit score input."""
    credit_score = extract_number(state["user_input"], "credit score")

    if credit_score is not None:
        credit_score = int(credit_score)
        state["credit_score"] = credit_score

        if credit_score < 620:
            state["final_response"] = "Score below 620 is considered subprime and is much harder to approve. You should improve your credit score first, but if you want to continue the process, we can. Do you want to continue? (yes/no)"
            state["application_step"] = "subprime_confirmation"
        else:
            state["final_response"] = "Great! What is the estimated home value/price?"
            state["application_step"] = "home_value"
    else:
        state["final_response"] = "Please enter a valid credit score (numeric value)."
        state["application_step"] = "credit_score"

    return state

def process_subprime_confirmation(state: AgentState) -> AgentState:
    """Process subprime continuation confirmation."""
    user_response = state["user_input"].strip().lower()

    if user_response in ["yes", "y", "continue", "proceed"]:
        state["subprime_continue"] = True
        state["final_response"] = "Understood. Let's continue. What is the estimated home value/price?"
        state["application_step"] = "home_value"
    else:
        state["subprime_continue"] = False
        state["final_response"] = "I understand. I'd recommend working on improving your credit score before applying. Feel free to come back when you're ready!"
        state["application_step"] = "ended"

    return state

def process_home_value(state: AgentState) -> AgentState:
    """Process home value input."""
    home_value = extract_number(state["user_input"], "home value")

    if home_value is not None:
        state["home_value"] = int(home_value)
        state["final_response"] = "How much down payment can you make?"
        state["application_step"] = "down_payment"
    else:
        state["final_response"] = "Please enter a valid home value (numeric value)."
        state["application_step"] = "home_value"

    return state

def process_down_payment(state: AgentState) -> AgentState:
    """Process down payment input (can be dollar amount or percentage)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data extraction assistant. Extract the down payment from the user's message.

        The user may provide:
        1. A dollar amount (e.g., "$50,000", "50000", "fifty thousand")
        2. A percentage (e.g., "10%", "10 percent", "ten percent")

        Return ONLY in this format:
        - If dollar amount: "AMOUNT:50000"
        - If percentage: "PERCENT:10"
        - If cannot determine: "NONE"

        Examples:
        - "I can put down $50,000" -> AMOUNT:50000
        - "10%" -> PERCENT:10
        - "I want to put down 20 percent" -> PERCENT:20
        - "I don't know" -> NONE"""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})
    response = result.content.strip()

    down_payment_amount = None

    if response.startswith("AMOUNT:"):
        try:
            down_payment_amount = int(float(response.split(":")[1]))
        except:
            pass
    elif response.startswith("PERCENT:"):
        try:
            percentage = float(response.split(":")[1])
            down_payment_amount = int((percentage / 100) * state["home_value"])
        except:
            pass

    if down_payment_amount is not None:
        state["down_payment"] = down_payment_amount
        # Calculate loan amount
        state["loan_amount"] = state["home_value"] - down_payment_amount
        state["final_response"] = "What is your annual income?"
        state["application_step"] = "income"
    else:
        state["final_response"] = "Please enter a valid down payment amount (either a dollar amount or percentage)."
        state["application_step"] = "down_payment"

    return state

def process_income(state: AgentState) -> AgentState:
    """Process annual income input."""
    income = extract_number(state["user_input"], "annual income")

    if income is not None:
        state["income"] = int(income)
        state["final_response"] = "What are your total monthly debt payments?"
        state["application_step"] = "debts"
    else:
        state["final_response"] = "Please enter a valid annual income (numeric value)."
        state["application_step"] = "income"

    return state

def process_debts(state: AgentState) -> AgentState:
    """Process monthly debts input."""
    debts = extract_number(state["user_input"], "monthly debt payment")

    if debts is not None:
        state["debts"] = debts
        state["final_response"] = "What loan term do you prefer? (15 or 30 years)"
        state["application_step"] = "loan_term"
    else:
        state["final_response"] = "Please enter a valid monthly debt amount (numeric value)."
        state["application_step"] = "debts"

    return state

def process_loan_term(state: AgentState) -> AgentState:
    """Process loan term input."""
    user_input = state["user_input"].strip()

    if "15" in user_input:
        state["loan_term"] = 15
    elif "30" in user_input:
        state["loan_term"] = 30
    else:
        state["final_response"] = "Please choose either 15 or 30 years."
        state["application_step"] = "loan_term"
        return state

    ltv = (state["loan_amount"] / state["home_value"]) * 100
    monthly_income = state["income"] / 12
    dti = (state["debts"] / monthly_income) * 100

    state["final_response"] = f"Got it! We can see if you will be approved or not.\n\nCredit Score: {state['credit_score']}\n\nLoan Amount: {state['loan_amount']:,}\n\nHome Value: {state['home_value']:,}\n\nLTV: {ltv:.1f}%\n\nDTI: {dti:.1f}%\n\nLoan Term: {state['loan_term']} years\n\nWould you like to see your rate?"
    state["application_step"] = "calculate_rate"

    return state

def calculate_rate(state: AgentState) -> AgentState:
    """Calculate the interest rate based on the collected application data."""
    # Calculate LTV and DTI
    ltv = (state["loan_amount"] / state["home_value"]) * 100
    monthly_income = state["income"] / 12
    dti = (state["debts"] / monthly_income) * 100

    # Use rate calculation tool
    input_data = f"{state['credit_score']},{ltv:.1f},{dti:.1f},{state['loan_term']}"
    tool_result = rate_calculation_tool.run(input_data)

    rate = None
    rate_match = re.search(r'(\d+\.\d+)%', tool_result)
    if rate_match:
        rate = float(rate_match.group(1))
        state["calculated_rate"] = rate

    summary = f"Thank you! Based on your information:\n\nCredit Score: {state['credit_score']}\nLoan Amount: {state['loan_amount']:,}\nHome Value: ${state['home_value']:,}\nLTV: {ltv:.1f}%\nDTI: {dti:.1f}%\nLoan Term: {state['loan_term']} years"

    if rate:
        state["final_response"] = f"{summary}\n\n{tool_result} Let me know if you have any other questions!"
    else:
        state["final_response"] = f"{summary}\n\n{tool_result}"

    state["application_step"] = "ended"
    return state

def route_after_validation(state: AgentState) -> str:
    """Route after topic validation."""
    if state["mode"] == "error":
        return "error"
    return "valid"

def check_app_step(state: AgentState) -> AgentState:
    """Pass-through node to check application step."""
    return state

def route_by_app_step(state: AgentState) -> str:
    """Route based on current application step."""
    app_step = state.get("application_step")

    if app_step == "credit_score":
        return "process_credit_score"
    elif app_step == "subprime_confirmation":
        return "process_subprime_confirmation"
    elif app_step == "home_value":
        return "process_home_value"
    elif app_step == "down_payment":
        return "process_down_payment"
    elif app_step == "income":
        return "process_income"
    elif app_step == "debts":
        return "process_debts"
    elif app_step == "loan_term":
        return "process_loan_term"
    elif app_step == "calculate_rate":
        return "calculate_rate"
    else:
        return "route_intent"

def route_mode(state: AgentState) -> str:
    """Route to appropriate node based on mode."""
    return state["mode"]

def create_workflow():
    """Create and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("validate_topic", validate_topic)
    workflow.add_node("check_app_step", check_app_step)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("start_application", start_application)
    workflow.add_node("process_credit_score", process_credit_score)
    workflow.add_node("process_subprime_confirmation", process_subprime_confirmation)
    workflow.add_node("process_home_value", process_home_value)
    workflow.add_node("process_down_payment", process_down_payment)
    workflow.add_node("process_income", process_income)
    workflow.add_node("process_debts", process_debts)
    workflow.add_node("process_loan_term", process_loan_term)
    workflow.add_node("calculate_rate", calculate_rate)

    # Define edges
    workflow.set_entry_point("check_app_step")

    workflow.add_conditional_edges(
        "check_app_step",
        route_by_app_step,
        {
            "route_intent": "validate_topic",
            "process_credit_score": "process_credit_score",
            "process_subprime_confirmation": "process_subprime_confirmation",
            "process_home_value": "process_home_value",
            "process_down_payment": "process_down_payment",
            "process_income": "process_income",
            "process_debts": "process_debts",
            "process_loan_term": "process_loan_term",
            "calculate_rate": "calculate_rate",
        }
    )

    workflow.add_conditional_edges(
        "validate_topic",
        route_after_validation,
        {
            "error": END,
            "valid": "route_intent"
        }
    )

    workflow.add_conditional_edges(
        "route_intent",
        route_mode,
        {
            "qa": "retrieve_documents",
            "application": "start_application"
        }
    )

    workflow.add_edge("retrieve_documents", "check_relevance")
    workflow.add_edge("check_relevance", "answer_question")
    workflow.add_edge("answer_question", END)
    workflow.add_edge("start_application", END)
    workflow.add_edge("process_credit_score", END)
    workflow.add_edge("process_subprime_confirmation", END)
    workflow.add_edge("process_home_value", END)
    workflow.add_edge("process_down_payment", END)
    workflow.add_edge("process_income", END)
    workflow.add_edge("process_debts", END)
    workflow.add_edge("process_loan_term", END)
    workflow.add_edge("calculate_rate", END)

    return workflow.compile()
