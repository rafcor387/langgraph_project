from langchain_core.tools import tool
from tools.weather_tools import classify_weather_pattern
from datetime import date as date_type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json    
import os


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
    lanzamiento. Se usa para consultar, inspeccionar o resumir los datos del
    perfil. No clasifica la estabilidad atmosférica; para esa intención se usa
    classify_radiosonde_stability.

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
    un resumen y puede ser "perfil mixto". Se usa cuando el usuario pide una
    clasificación o interpretación de estabilidad, no para una consulta
    general de los datos. No usa el modelo LSTM temporal.

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
def analyze_wind(profile_id: int) -> dict:
    """Analiza el perfil vertical de viento de un radiosondeo con MetPy.

    Usa el profile_id devuelto por search_radiosondes. Devuelve viento de
    superficie y máximo, viento medio y cizalladura 0–1/0–3/0–6 km AGL,
    movimiento de tormenta Bunkers y helicidad relativa a la tormenta cuando
    la cobertura lo permite. No genera una imagen; para eso usa
    generate_hodograph.

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
    url = f"{api_base_url}/feature/radiosondes/{normalized_profile_id}/wind/"
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
            "error": f"No se pudo analizar el viento: {exc}",
            "profile_id": normalized_profile_id,
        }

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("surface_wind"), dict)
        or not isinstance(payload.get("layers"), dict)
    ):
        return {
            "error": "El servicio devolvió un análisis de viento inválido.",
            "profile_id": normalized_profile_id,
        }

    return payload

@tool
def generate_skew_t(profile_id: int) -> dict:
    """Genera y publica el descriptor de un diagrama Skew-T.

    Usa el profile_id devuelto por search_radiosondes. Django recupera y
    normaliza el TSV, genera el PNG completo con MetPy, lo cachea de forma
    privada en R2 y devuelve rutas relativas para visualizarlo o descargarlo,
    además de diagnósticos numéricos SB/ML/MU CAPE-CIN, LCL, LFC, EL, CCL,
    Lifted Index, agua precipitable y temperatura convectiva. La herramienta
    nunca incluye la imagen como Base64 ni expone el bucket u object_key.

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
    url = f"{api_base_url}/feature/radiosondes/{normalized_profile_id}/skew-t/"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=60) as response:
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
            "error": f"No se pudo generar el Skew-T: {exc}",
            "profile_id": normalized_profile_id,
        }

    if (
        not isinstance(payload, dict)
        or payload.get("type") != "skew_t_diagram"
        or not isinstance(payload.get("image_path"), str)
    ):
        return {
            "error": "El servicio devolvió un descriptor Skew-T inválido.",
            "profile_id": normalized_profile_id,
        }

    return payload


@tool
def generate_hodograph(profile_id: int) -> dict:
    """Genera y publica el descriptor de un hodógrafo.

    Usa el profile_id devuelto por search_radiosondes. Django recupera y
    normaliza el TSV, genera el PNG con MetPy, lo cachea de forma privada en R2
    y devuelve rutas relativas para visualizarlo o descargarlo. Incluye
    diagnósticos de viento en superficie, viento máximo y cizalladura vectorial
    0–1, 0–3 y 0–6 km AGL. No expone Base64, bucket ni object_key.

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
    url = f"{api_base_url}/feature/radiosondes/{normalized_profile_id}/hodograph/"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=60) as response:
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
            "error": f"No se pudo generar el hodógrafo: {exc}",
            "profile_id": normalized_profile_id,
        }

    if (
        not isinstance(payload, dict)
        or payload.get("type") != "hodograph_diagram"
        or not isinstance(payload.get("image_path"), str)
    ):
        return {
            "error": "El servicio devolvió un descriptor de hodógrafo inválido.",
            "profile_id": normalized_profile_id,
        }

    return payload

tools = [
    search_radiosondes,
    analyze_radiosonde,
    classify_radiosonde_stability,
    analyze_wind,
    classify_weather_pattern,
    generate_skew_t,
    generate_hodograph,
]
