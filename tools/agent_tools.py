from langchain_core.tools import tool
from tools.weather_tools import classify_weather_pattern
from datetime import date as date_type, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json    
import os


@tool
def search_radiosondes(
    date: str,
    time: str | None = None,
) -> dict:
    """Busca un radiosondeo de La Paz para una fecha exacta.

    Args:
        date: Fecha exacta YYYY-MM-DD.
        time: Hora UTC opcional HH:MMZ, por ejemplo 12:00Z. Si se omite, el
            backend devuelve el primer lanzamiento del día.

    Returns:
        count y radiosondes, con cero o una coincidencia. El perfil contiene
        profile_id, date, time y observed_at. Sólo consulta el catálogo; no
        descarga archivos TSV desde R2.
    """
    try:
        normalized_date = date_type.fromisoformat(date).isoformat()
    except (TypeError, ValueError):
        return {
            "error": "date debe tener el formato YYYY-MM-DD.",
            "count": 0,
            "radiosondes": [],
        }

    normalized_time = None
    if time is not None:
        try:
            normalized_time = datetime.strptime(time, "%H:%MZ").strftime("%H:%MZ")
        except (TypeError, ValueError):
            return {
                "error": "time debe tener el formato HH:MMZ, por ejemplo 12:00Z.",
                "count": 0,
                "radiosondes": [],
            }

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    query_params = {"date": normalized_date}
    if normalized_time is not None:
        query_params["time"] = normalized_time
    query = urlencode(query_params)
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


@tool
def generate_radiosonde_report(
    start_date: str,
    end_date: str,
    time: str | None = None,
) -> dict:
    """Solicita un informe PDF de radiosondeos para un intervalo inclusivo.

    Django localiza y procesa los perfiles secuencialmente, genera estadísticas
    y gráficos, guarda el PDF privado en R2 y devuelve un descriptor de estado.
    La generación es asíncrona: el estado inicial normalmente es processing y
    el frontend consulta status_path hasta que el informe queda ready.

    Args:
        start_date: Fecha inicial inclusiva YYYY-MM-DD.
        end_date: Fecha final inclusiva YYYY-MM-DD, hasta 366 días.
        time: Hora UTC opcional HH:MMZ para filtrar un ciclo concreto.
    """
    try:
        start = date_type.fromisoformat(start_date)
        end = date_type.fromisoformat(end_date)
    except (TypeError, ValueError):
        return {"error": "start_date y end_date deben usar el formato YYYY-MM-DD."}
    if end < start:
        return {"error": "end_date no puede ser anterior a start_date."}
    if (end - start).days + 1 > 366:
        return {"error": "La primera versión admite intervalos de hasta 366 días."}

    normalized_time = None
    if time is not None:
        try:
            normalized_time = datetime.strptime(time, "%H:%MZ").strftime("%H:%MZ")
        except (TypeError, ValueError):
            return {"error": "time debe tener el formato HH:MMZ, por ejemplo 12:00Z."}

    body = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if normalized_time is not None:
        body["time"] = normalized_time

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    url = f"{api_base_url}/feature/radiosonde-reports/"
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return _http_tool_error(exc, "generar el informe")
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"error": f"No se pudo solicitar el informe: {exc}"}

    if (
        not isinstance(payload, dict)
        or payload.get("type") != "radiosonde_report"
        or not isinstance(payload.get("report_id"), int)
        or not isinstance(payload.get("status_path"), str)
    ):
        return {"error": "El servicio devolvió un descriptor de informe inválido."}
    return payload


@tool
def get_radiosonde_report_status(report_id: int) -> dict:
    """Consulta el progreso de un informe PDF solicitado anteriormente.

    Args:
        report_id: Identificador entero positivo devuelto al crear el informe.
    """
    normalized_report_id = _positive_integer(report_id)
    if normalized_report_id is None:
        return {"error": "report_id debe ser un número entero positivo."}

    api_base_url = os.getenv("RADIOSONDE_API_URL", "http://localhost:8000").rstrip("/")
    url = f"{api_base_url}/feature/radiosonde-reports/{normalized_report_id}/"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return _http_tool_error(exc, "consultar el informe")
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"error": f"No se pudo consultar el informe: {exc}"}
    if not isinstance(payload, dict) or payload.get("type") != "radiosonde_report":
        return {"error": "El servicio devolvió un estado de informe inválido."}
    return payload


def _positive_integer(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0 or str(value).strip() != str(normalized):
        return None
    return normalized


def _http_tool_error(exc: HTTPError, action: str) -> dict:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        detail = payload.get("detail", payload)
    except Exception:
        detail = str(exc)
    return {
        "error": f"No se pudo {action}: {detail}",
        "status_code": exc.code,
    }

tools = [
    search_radiosondes,
    analyze_radiosonde,
    classify_radiosonde_stability,
    analyze_wind,
    classify_weather_pattern,
    generate_skew_t,
    generate_hodograph,
    generate_radiosonde_report,
    get_radiosonde_report_status,
]
