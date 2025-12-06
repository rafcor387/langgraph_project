import json
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from tools.agent_tools import tools
from config.openai import llm
from prompt.general_prompt import sys_msg

llm_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    messages_for_llm = []
    
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                content_data = json.loads(msg.content)
                
                if "image_base64" in content_data:
                    clean_content = json.dumps({
                        "type": content_data.get("type"),
                        "summary": content_data.get("summary"),
                    })
                    
                    clean_msg = ToolMessage(
                        content=clean_content,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name
                    )
                    messages_for_llm.append(clean_msg)
                else:
                    messages_for_llm.append(msg)
                    
            except (json.JSONDecodeError, TypeError):
                messages_for_llm.append(msg)
        else:
            messages_for_llm.append(msg)


    response = llm_tools.invoke([sys_msg] + messages_for_llm)
    
    return {"messages": [response]}