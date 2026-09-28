from langchain_core.tools import tool
#from utils.get_radiosonde import get_radiosonde_fromDB
from utils.calculations import calculos
from tools.weather_tools import classify_weather_pattern
from datetime import date as date_type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd
import json    
import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.calc import parcel_profile, lcl, lfc, el
from metpy.units import units

matplotlib.use('Agg')


@tool
def search_radiosondes(date: str) -> dict:
    """Busca los radiosondeos de La Paz disponibles para una fecha.

    Args:
        date: Fecha exacta en formato YYYY-MM-DD, por ejemplo 2018-05-14.

    Returns:
        Un objeto con count y una lista radiosondes. Cada elemento contiene
        profile_id, date, time y observed_at. Esta herramienta no descarga el
        archivo TSV.
    """
    try:
        normalized_date = date_type.fromisoformat(date).isoformat()
    except (TypeError, ValueError):
        return {
            "error": "La fecha debe tener el formato YYYY-MM-DD.",
            "count": 0,
            "radiosondes": [],
        }

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    query = urlencode({"date": normalized_date})
    url = f"{api_base_url}/feature/radiosondes/search/?{query}"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return {
            "error": f"El servicio de radiosondeos respondió HTTP {exc.code}: {detail}",
            "count": 0,
            "radiosondes": [],
        }
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "error": f"No se pudo consultar el servicio de radiosondeos: {exc}",
            "count": 0,
            "radiosondes": [],
        }

    radiosondes = payload.get("radiosondes")
    if not isinstance(radiosondes, list):
        return {
            "error": "El servicio devolvió una respuesta inválida.",
            "count": 0,
            "radiosondes": [],
        }

    # Se filtra explícitamente la salida para no exponer bucket u object_key.
    compact_results = [
        {
            "profile_id": item.get("profile_id"),
            "date": item.get("date"),
            "time": item.get("time"),
            "observed_at": item.get("observed_at"),
        }
        for item in radiosondes
    ]
    return {
        "count": len(compact_results),
        "radiosondes": compact_results,
    }


@tool
def analyze_radiosonde(profile_id: int) -> dict:
    """Obtiene el resumen general normalizado de un radiosondeo.

    Primero debe obtenerse el profile_id con search_radiosondes. El backend
    recupera el TSV desde Cloudflare R2, lo normaliza en memoria y devuelve
    metadatos, control de calidad, cobertura, superficie, tope y ubicación de
    lanzamiento. Esta herramienta todavía no calcula índices termodinámicos
    de MetPy como CAPE, CIN, LCL, LFC o EL.

    Args:
        profile_id: Identificador entero positivo del radiosondeo.
    """
    if isinstance(profile_id, bool):
        return {"error": "profile_id debe ser un número entero positivo."}

    try:
        normalized_profile_id = int(profile_id)
    except (TypeError, ValueError):
        return {"error": "profile_id debe ser un número entero positivo."}

    if normalized_profile_id <= 0 or str(profile_id).strip() != str(normalized_profile_id):
        return {"error": "profile_id debe ser un número entero positivo."}

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    url = f"{api_base_url}/feature/radiosondes/{normalized_profile_id}/profile/"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = error_payload.get(
                "error",
                error_payload.get("detail", str(error_payload)),
            )
        except Exception:
            detail = str(exc)
        return {
            "error": detail,
            "status_code": exc.code,
            "profile_id": normalized_profile_id,
        }
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "error": f"No se pudo analizar el radiosondeo: {exc}",
            "profile_id": normalized_profile_id,
        }

    if not isinstance(payload, dict) or not isinstance(payload.get("profile"), dict):
        return {
            "error": "El servicio devolvió una respuesta de análisis inválida.",
            "profile_id": normalized_profile_id,
        }

    return payload


@tool
def classify_radiosonde_stability(profile_id: int) -> dict:
    """Diagnostica por capas la estabilidad de un radiosondeo de La Paz.

    Usa el profile_id devuelto por search_radiosondes. El backend descarga el
    TSV desde R2, lo normaliza y calcula con MetPy CAPE/CIN de superficie y de
    capa mezclada, gradientes térmicos, N² e inversión superficial. Después
    devuelve ejes separados de estabilidad estática, estabilidad de parcela,
    potencial convectivo e inversión superficial. La categoría global es sólo
    un resumen y puede ser "perfil mixto". No usa el modelo LSTM temporal.

    Args:
        profile_id: Identificador entero positivo del radiosondeo.
    """
    if isinstance(profile_id, bool):
        return {"error": "profile_id debe ser un número entero positivo."}

    try:
        normalized_profile_id = int(profile_id)
    except (TypeError, ValueError):
        return {"error": "profile_id debe ser un número entero positivo."}

    if normalized_profile_id <= 0 or str(profile_id).strip() != str(normalized_profile_id):
        return {"error": "profile_id debe ser un número entero positivo."}

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    url = f"{api_base_url}/feature/radiosondes/{normalized_profile_id}/stability/"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = error_payload.get(
                "error",
                error_payload.get("detail", str(error_payload)),
            )
        except Exception:
            detail = str(exc)
        return {
            "error": detail,
            "status_code": exc.code,
            "profile_id": normalized_profile_id,
        }
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "error": f"No se pudo clasificar el radiosondeo: {exc}",
            "profile_id": normalized_profile_id,
        }

    if not isinstance(payload, dict) or not isinstance(
        payload.get("classification"),
        dict,
    ):
        return {
            "error": "El servicio devolvió una clasificación inválida.",
            "profile_id": normalized_profile_id,
        }

    return payload

@tool
def diagram_skew_t(fecha: str):
    """
    Genera un diagrama Skew-T visual a partir de datos de radiosondeo.
    
    Args:
        fecha (str): La fecha del radiosondeo en formato estricto 'YYYY-MM-DD' 
                     (Ejemplo: '2024-04-03'). No incluir la hora.
    """
    # --- PASO 1: Obtener el radiosondeo ---
    nombre_archivo = f"{fecha}-12Z.csv"
    
    # Usamos tu ruta corregida
    ruta_archivo = f"../my_lstm_agent/radiosonde/{nombre_archivo}"

    print(f"DEBUG: Buscando archivo en: {ruta_archivo}") 

    if not os.path.exists(ruta_archivo):
        return f"Error: No se encontró el archivo de radiosondeo en la ruta {ruta_archivo}. Verifica la fecha."

    # --- PASO 2: Generar el gráfico con MetPy ---
    try:
        df = pd.read_csv(ruta_archivo)
        df.columns = df.columns.str.strip()

        p = df['pressure_hPa'].values * units.hPa
        t = df['temp_C'].values * units.degC 
        td = df['dewpoint_C'].values * units.degC

        prof = parcel_profile(p, t[0], td[0])
        
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig)

        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        skew.plot(p, t, 'red', label="Temperatura")
        skew.plot(p, td, 'green', label="Punto de Rocío")
        skew.plot(p, prof.to('degC'), 'black', label='Parcela')

        skew.shade_cin(p, t, prof)
        skew.shade_cape(p, t, prof)

        lcl_pressure, lcl_temperature = lcl(p[0], t[0], td[0])
        skew.ax.plot(lcl_temperature, lcl_pressure, 'ko', markerfacecolor='cyan', label='LCL')

        try:
            lfc_pressure, lfc_temperature = lfc(p, t, td)
            el_pressure, el_temperature = el(p, t, td)
            if lfc_pressure: 
                skew.ax.plot(lfc_temperature, lfc_pressure, 'ko', markerfacecolor='magenta', label='LFC')
            if el_pressure: 
                skew.ax.plot(el_temperature, el_pressure, 'ko', markerfacecolor='orange', label='EL')
        except:
            pass 

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 40)
        plt.title(f"Diagrama Skew-T: {fecha} (12Z)")
        plt.legend()
        
        # --- PASO 3: CONVERTIR A BASE64 Y RETORNAR JSON ---
        
        # 1. Crear buffer en memoria
        buf = io.BytesIO()
        
        # 2. Guardar figura en el buffer
        # bbox_inches='tight' recorta los bordes blancos sobrantes para que se vea mejor en web
        fig.savefig(buf, format='png', bbox_inches='tight')
        
        # 3. Volver al inicio del buffer
        buf.seek(0)
        
        # 4. Codificar a Base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # 5. Limpiar memoria (Vital para no saturar el servidor)
        plt.close(fig)
        buf.close()

        # 6. Crear estructura JSON para el Frontend
        # Esto es lo que leerá React para saber que tiene que pintar una imagen
        respuesta_json = {
            "type": "skew_t_diagram", 
            "image_base64": f"data:image/png;base64,{img_str}",
        }

        # Retornamos el JSON como string
        return json.dumps(respuesta_json)

    except Exception as e:
        return f"Error procesando los datos del archivo {nombre_archivo}: {str(e)}"

tools = [
    search_radiosondes,
    analyze_radiosonde,
    classify_radiosonde_stability,
    classify_weather_pattern,
    diagram_skew_t,
]
