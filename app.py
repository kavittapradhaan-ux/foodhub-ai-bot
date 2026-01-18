import streamlit as st
from llm_setup import FoodHubMistral
from sql_agent import get_sql_agent
from chat_agent import build_final_agent

st.set_page_config(page_title="FoodHub AI Support", page_icon="🍔")
st.title("🍔 FoodHub Customer Support Bot")

@st.cache_resource
def load_foodhub_system():
    # Initialize the modular system
    brain = FoodHubMistral(model_path="finetuned_mistral_llm")
    sql_executor = get_sql_agent(brain)
    return build_final_agent(brain, sql_executor)

# System Initialization
with st.spinner("Waking up the Expert AI Brain..."):
    agent = load_foodhub_system()

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_input := st.chat_input("How can I help with your order?"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = agent.invoke({"input": user_input})
        answer = response["output"]
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
