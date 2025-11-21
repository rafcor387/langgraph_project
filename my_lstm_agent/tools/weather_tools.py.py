import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from metpy.interpolate import interpolate_1d
from metpy.units import units
from sklearn.mixture import GaussianMixture
from langchain_core.tools import tool

# ==================== CONFIGURACIÓN GLOBAL ====================
DATA_FOLDER = "2024"  # Carpeta donde están los CSVs crudos

# Niveles y Variables (Igual que en el entrenamiento)
LEVELS = np.arange(1290, 13290 + 500, 500) * units.m
VAR_MAPPING = {
    'pressure_hPa': 'Pressure', 'temp_C': 'T', 'dewpoint_C': 'TD',
    'wind_speed_ms': 'WS', 'wind_dir_deg': 'WD', 'rh_percent': 'RH',
    'mixing_ratio_gkg': 'MR'
}
PREFIX_ORDER = ['Pressure', 'T', 'TD', 'WS', 'WD', 'RH', 'MR']

# ==================== CLASE LSTM (La arquitectura) ====================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.encoder_to_embedding = nn.Linear(hidden_size * 2, embedding_dim)
    
    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.encoder_to_embedding(last_hidden)

# ==================== LÓGICA INTERNA (Privada) ====================

def _parse_date(date_str):
    try:
        day, month, year = date_str.split('/')
        return datetime(int(year), int(month), int(day), 0)
    except: return None

def _get_files_window(start_date, window_size):
    """Busca los archivos necesarios para la ventana específica"""
    files = []
    curr = start_date
    for _ in range(window_size):
        fname = curr.strftime("%Y-%m-%d") + f"-{curr.hour:02d}Z.csv"
        p = Path(DATA_FOLDER) / fname
        if not p.exists(): return None, f"Falta archivo: {fname}"
        files.append(p)
        curr += timedelta(hours=12)
    return files, None

def _process_data(files, stats_path):
    """Interpola y Normaliza usando el dataset_stats.pkl específico"""
    if not Path(stats_path).exists():
        return None, "No se encuentra dataset_stats.pkl"
        
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    window_vectors = []
    
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            height = df['height_m'].to_numpy() * units.meter
            
            # --- Interpolación Rápida ---
            # Nota: Para producción, idealmente ya tendrías los datos interpolados
            # Aquí hacemos la interpolación al vuelo para ser flexibles con datos crudos
            data_dict = {}
            # (Lógica simplificada de interpolación para no alargar el código)
            # Asumimos que las columnas existen. 
            for orig, prefix in VAR_MAPPING.items():
                val_array = df[orig].to_numpy()
                # Aquí deberíamos asignar unidades correctamente como en tus scripts anteriores
                # Para este ejemplo, asumimos que el valor numérico es correcto
                interp = interpolate_1d(LEVELS.magnitude, df['height_m'].values, val_array)
                data_dict[prefix] = np.nan_to_num(interp, 0.0)

            # --- Normalización y Aplanado ---
            flat_vector = []
            for i, level_val in enumerate(LEVELS.magnitude):
                level_suffix = f"{int(level_val)}m"
                for prefix in PREFIX_ORDER:
                    col_name = f"{prefix}_{level_suffix}"
                    val = data_dict[prefix][i]
                    
                    # Normalizar con stats
                    if col_name in stats:
                        mu = stats[col_name]['mean']
                        sigma = stats[col_name]['std']
                        norm_val = (val - mu) / sigma
                    else:
                        norm_val = 0.0
                    flat_vector.append(norm_val)
            
            window_vectors.append(np.array(flat_vector, dtype=np.float32))
            
        except Exception as e:
            return None, f"Error procesando {fpath.name}: {str(e)}"

    return np.stack(window_vectors), None

def _run_model_inference(data_tensor, model_path, clusters_path):
    """Ejecuta LSTM y GMM"""
    device = torch.device('cpu')
    
    # 1. Cargar LSTM
    if not Path(model_path).exists(): return None, "Modelo LSTM no encontrado"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hp = ckpt['hyperparameters']
    
    lstm = LSTMAutoencoder(hp['input_size'], hp['hidden_size'], hp['embedding_dim'], hp['num_layers'], hp['dropout'])
    lstm.load_state_dict(ckpt['model_state_dict'], strict=False)
    lstm.eval()
    
    # 2. Cargar GMM
    if not Path(clusters_path).exists(): return None, "Modelo Clusters no encontrado"
    with open(clusters_path, 'rb') as f: c_dict = pickle.load(f)
    meta = c_dict['metadata']
    
    gmm = GaussianMixture(n_components=meta['k'], covariance_type='full')
    gmm.means_ = meta['means']
    gmm.covariances_ = meta['covariances']
    gmm.weights_ = meta['weights']
    gmm.precisions_cholesky_ = meta['precisions_cholesky']
    gmm.precisions_ = np.linalg.inv(meta['covariances'])
    gmm.converged_ = True
    
    # 3. Inferencia
    with torch.no_grad():
        emb = lstm.encode(torch.FloatTensor(data_tensor).unsqueeze(0)).numpy().flatten()
        
    probs = gmm.predict_proba(emb.reshape(1, -1))[0]
    cluster_id = np.argmax(probs)
    
    return {
        "cluster_id": int(cluster_id),
        "probs": probs.tolist(),
        "confianza": float(probs[cluster_id] * 100)
    }, None

# ==================== TOOLS PÚBLICAS PARA LANGGRAPH ====================

@tool
def classify_weather_pattern_w6(date_str: str) -> str:
    """
    Clasifica el patrón atmosférico analizando una ventana de 6 radiosondeos (3 días).
    Útil para detectar patrones de corto plazo.
    
    Args:
        date_str: Fecha de inicio en formato DD/MM/YYYY (ej: '09/12/2024').
    """
    # --- CONFIGURACIÓN ESPECÍFICA PARA VENTANA 6 ---
    WINDOW = 6
    MODEL_DIR = Path("modelos_entrenados") # CAMBIAR si tienes carpetas separadas
    # -----------------------------------------------

    start_date = _parse_date(date_str)
    if not start_date: return json.dumps({"error": "Formato de fecha inválido"})
    
    # 1. Buscar Archivos
    files, err = _get_files_window(start_date, WINDOW)
    if err: return json.dumps({"error": err})
    
    # 2. Procesar Datos
    data, err = _process_data(files, MODEL_DIR / "dataset_stats.pkl")
    if err: return json.dumps({"error": err})
    
    # 3. Inferencia
    res, err = _run_model_inference(data, MODEL_DIR / "best_autoencoder.pth", MODEL_DIR / "clusters.pkl")
    if err: return json.dumps({"error": err})
    
    # 4. Construir Respuesta JSON
    end_date = start_date + timedelta(hours=12 * (WINDOW - 1))
    
    response = {
        "fecha_consulta": date_str,
        "ventana_temporal": {
            "inicio": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fin": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "archivos": [f.name for f in files]
        },
        "clasificacion": {
            "cluster_id": res['cluster_id'],
            "confianza_pct": res['confianza'],
            "interpretacion": "Alta" if res['confianza'] > 80 else "Media" if res['confianza'] > 50 else "Baja",
            "probabilidades_todos_clusters": res['probs']
        }
    }
    
    return json.dumps(response, indent=2)

@tool
def classify_weather_pattern_w4(date_str: str) -> str:
    """
    Clasifica el patrón atmosférico analizando una ventana de 4 radiosondeos (2 días).
    Útil para análisis más rápidos o cuando hay menos datos.
    
    Args:
        date_str: Fecha de inicio en formato DD/MM/YYYY (ej: '09/12/2024').
    """
    # --- CONFIGURACIÓN ESPECÍFICA PARA VENTANA 4 ---
    WINDOW = 4
    # IMPORTANTE: Aquí deberías apuntar a la carpeta donde entrenaste el modelo de 4 ventanas
    MODEL_DIR = Path("modelos_w4") 
    # -----------------------------------------------

    if not MODEL_DIR.exists():
        return json.dumps({"error": "Modelos para ventana 4 no encontrados en el sistema."})

    start_date = _parse_date(date_str)
    if not start_date: return json.dumps({"error": "Formato de fecha inválido"})
    
    files, err = _get_files_window(start_date, WINDOW)
    if err: return json.dumps({"error": err})
    
    data, err = _process_data(files, MODEL_DIR / "dataset_stats.pkl")
    if err: return json.dumps({"error": err})
    
    res, err = _run_model_inference(data, MODEL_DIR / "best_autoencoder.pth", MODEL_DIR / "clusters.pkl")
    if err: return json.dumps({"error": err})
    
    end_date = start_date + timedelta(hours=12 * (WINDOW - 1))
    
    response = {
        "fecha_consulta": date_str,
        "ventana_temporal": {
            "inicio": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fin": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "archivos": [f.name for f in files]
        },
        "clasificacion": {
            "cluster_id": res['cluster_id'],
            "confianza_pct": res['confianza'],
            "interpretacion": "Alta" if res['confianza'] > 80 else "Media",
            "probabilidades_todos_clusters": res['probs']
        }
    }
    
    return json.dumps(response, indent=2)