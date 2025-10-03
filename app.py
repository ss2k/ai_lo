from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from workflow import create_workflow

st.set_page_config(page_title="AI Loan Officer", page_icon="🏦")

# session initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = {
        "user_input": "",
        "messages": [],
        "mode": "qa",
        "intent": None,
        "retrieved_docs": None,
        "context": None,
        "credit_score": None,
        "home_value": None,
        "down_payment": None,
        "loan_amount": None,
        "income": None,
        "debts": None,
        "loan_term": None,
        "application_step": None,
        "subprime_continue": None,
        "final_response": None,
        "calculated_rate": None
    }

if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow()

# Title
st.title("🏦 AI Loan Officer")
st.caption("Ask questions about our loan products or start a mortgage application")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.session_state.workflow_state["user_input"] = prompt

    with st.spinner("Thinking..."):
        result = st.session_state.workflow.invoke(st.session_state.workflow_state)

    st.session_state.workflow_state = result

    response = result.get("final_response", "I'm sorry, I couldn't process that.")
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if result.get("application_step") == "ended":
        st.session_state.workflow_state["application_step"] = None
        st.session_state.workflow_state["mode"] = "qa"

    st.rerun()

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This AI Loan Officer can:")
    st.write("- Answer questions about loan products")
    st.write("- Help you start a mortgage application")
    st.write("- Calculate your interest rate and give you a preapproval")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.workflow_state = {
            "user_input": "",
            "messages": [],
            "mode": "qa",
            "intent": None,
            "retrieved_docs": None,
            "context": None,
            "credit_score": None,
            "home_value": None,
            "down_payment": None,
            "loan_amount": None,
            "income": None,
            "debts": None,
            "loan_term": None,
            "application_step": None,
            "subprime_continue": None,
            "final_response": None,
            "calculated_rate": None
        }
        st.rerun()
