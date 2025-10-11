import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

load_dotenv()

uri = os.getenv("MONGO_URI")

client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("¡Ping exitoso! Te has conectado correctamente a MongoDB.")
except Exception as e:
    print(f"Error de conexión: {e}")
    exit()

db = client['test']       
collection = db['radiosondeo']  
