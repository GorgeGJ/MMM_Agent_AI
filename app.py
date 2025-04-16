import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from agent.tools import get_historical_contribution, simulate_budget_scenario
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Page setup
st.set_page_config(page_title="MMM Agent AI", page_icon="📊")
st.title("📈 Marketing Mix Modeling Agent")
st.markdown("Ask questions like:\n- *'How much did Paid Social contribute in 2024?'\n- *'If I add $500K to YouTube in Q3, what would happen?'*")

# Input
user_input = st.text_input("Ask a marketing performance or budget scenario question:")

# On user query
if user_input:
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_key)
    agent = initialize_agent(
        tools=[get_historical_contribution, simulate_budget_scenario],
        llm=llm,
        agent="chat-zero-shot-react-description",
        verbose=True
    )
    with st.spinner("Thinking..."):
        try:
            response = agent.run(user_input)
            st.success(response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
