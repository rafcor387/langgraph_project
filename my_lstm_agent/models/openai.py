from langchain_groq import ChatGroq

# Define LLM with bound tools
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.8)