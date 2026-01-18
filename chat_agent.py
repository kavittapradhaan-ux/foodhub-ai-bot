from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

def build_final_agent(llm, sql_agent):
    # Tool 1: Order Query Tool with Security Guardrails
    def db_tool(query):
        if any(word in query.lower() for word in ["every", "all", "dump", "database"]):
            return "Access Denied: You are restricted from querying bulk data for security reasons."
        return sql_agent.invoke({"input": query})["output"]

    # Tool 2: Polite Answer Tool
    def response_formatter(data):
        return llm.invoke(f"Rewrite this raw order data into a friendly customer message: {data}")

    tools = [
        Tool(
            name="Order_Lookup", 
            func=db_tool, 
            description="Use this to fetch status and details for a specific Order ID."
        ),
        Tool(
            name="Polite_Formatter", 
            func=response_formatter, 
            description="Use this to finalize the response tone."
        )
    ]

    # Manual ReAct Prompt implementation for environment stability
    template = """Answer the customer query. Use the following format:
    Question: {input}
    Thought: Identify if I need to look up data.
    Action: [Order_Lookup] or [Polite_Formatter]
    Action Input: The specific query
    Observation: Result
    Thought: I have the answer.
    Final Answer: The polite final response.
    
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
