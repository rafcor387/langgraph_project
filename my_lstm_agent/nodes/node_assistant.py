from langgraph.graph import MessagesState
from tools.aritmetic_tools import tools
from models.openai import llm
from prompt.general_prompt import sys_msg

# Node
llm_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    return {"messages": [llm_tools.invoke([sys_msg] + state["messages"])]}

