import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()
if not GROQ_MODEL:
    raise RuntimeError(
        "Falta GROQ_MODEL en las variables de entorno. "
        "Configura un modelo válido de Groq antes de iniciar LangGraph."
    )

# Define LLM with bound tools
try:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.8)
except Exception as exc:
    raise RuntimeError(
        f"No se pudo configurar el modelo de Groq '{GROQ_MODEL}'. "
        "Verifica GROQ_MODEL y GROQ_API_KEY."
    ) from exc
