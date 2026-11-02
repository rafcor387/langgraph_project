"""Asignaciones aleatorias persistentes para demostrar el flujo de clusters."""

from contextlib import contextmanager
import csv
from datetime import date as date_type, datetime, timedelta, timezone
import os
from pathlib import Path
import secrets
import tempfile
import threading
from typing import Literal

from langchain_core.tools import tool


CSV_PATH = Path(__file__).resolve().parents[1] / "dataset" / "weather_pattern_assignments.csv"
CLUSTER_IDS = (1, 2, 3)  # Grupos de demostración, no IDs de los GMM entrenados.
WINDOW_SIZE = 6
FIELDS = ("date", "window", "cluster_id", "assignment_method", "created_at")
_THREAD_LOCK = threading.Lock()


@contextmanager
def _locked_dataset(path):
    """Serializa lectura/asignación/escritura entre hilos y procesos locales."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        with path.with_suffix(".csv.lock").open("a+b") as lock:
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("El CSV de asignaciones no tiene el encabezado esperado; no se modificó.")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(row.get(field) is None for field in FIELDS):
            raise ValueError("El CSV de asignaciones contiene una fila incompleta; no se modificó.")
        canonical_date = date_type.fromisoformat(row["date"]).isoformat()
        window, cluster = int(row["window"]), int(row["cluster_id"])
        if canonical_date != row["date"] or window != WINDOW_SIZE or cluster not in CLUSTER_IDS or row["assignment_method"] != "random":
            raise ValueError("El CSV contiene un registro incompatible con el formato esperado; no se modificó.")
        datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        key = (canonical_date, window)
        if key in seen:
            raise ValueError("El CSV contiene registros duplicados para una fecha y ventana; no se modificó.")
        seen.add(key)
    return rows


def _write_rows(path, rows):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _assignment(date, window, path):
    with _locked_dataset(path):
        rows = _read_rows(path)
        existing = next((row for row in rows if row["date"] == date and int(row["window"]) == window), None)
        if existing is not None:
            return existing, True
        row = {
            "date": date, "window": str(window),
            "cluster_id": str(secrets.choice(CLUSTER_IDS)),
            "assignment_method": "random",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        rows.append(row)
        _write_rows(path, rows)
        return row, False


@tool
def assign_weather_pattern(date: str, window: Literal[6] = 6) -> dict:
    """Obtiene el cluster asignado a una fecha para una ventana de 6 radiosondeos.

    Devuelve la fecha, los tres días de la ventana, el total de seis radiosondeos
    (dos por día) y el identificador del grupo asignado.
    Consultas repetidas para la misma fecha conservan el mismo identificador.
    La herramienta no proporciona interpretación física ni métricas de confianza.

    Args:
        date: Fecha inicial YYYY-MM-DD; también acepta DD/MM/YYYY.
        window: Ventana fija de 6 radiosondeos. El único valor permitido es 6.
    """
    try:
        start_date = (
            datetime.strptime(date, "%d/%m/%Y").date()
            if "/" in date else date_type.fromisoformat(date)
        )
        normalized_date = start_date.isoformat()
        window_dates = [(start_date + timedelta(days=offset)).isoformat() for offset in range(3)]
    except (TypeError, ValueError, OverflowError):
        return {"error": "Usa una fecha válida YYYY-MM-DD o DD/MM/YYYY."}
    if isinstance(window, bool) or not isinstance(window, int) or window != WINDOW_SIZE:
        return {"error": "Sólo está disponible la ventana de 6 radiosondeos."}
    try:
        row, cached = _assignment(normalized_date, window, CSV_PATH)
    except (OSError, ValueError, csv.Error) as exc:
        return {"error": f"No se pudo consultar o guardar la asignación: {exc}"}
    return {
        "type": "weather_pattern_assignment",
        "date": normalized_date,
        "window": window,
        "window_dates": window_dates,
        "radiosonde_count": WINDOW_SIZE,
        "radiosondes_per_day": 2,
        "cluster_id": int(row["cluster_id"]),
        "cached": cached,
        "created_at": row["created_at"],
        "summary": (
            f"Fecha: {normalized_date}\n"
            f"Días de la ventana: {', '.join(window_dates)}\n"
            f"Ventana: {window} radiosondeos\n"
            f"Cluster asignado: {row['cluster_id']}"
        ),
    }
