import sqlite3
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

def get_sql_agent(llm, db_path="customer_orders.db"):
    # Connect to the local SQLite database
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # Create the internal SQL executor
    return create_sql_agent(
        llm=llm, 
        db=db, 
        agent_type="tool-calling", 
        verbose=False
    )
