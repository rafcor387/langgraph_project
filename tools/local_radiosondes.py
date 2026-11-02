"""Lectura local de radiosondeos EDT sin depender de PostgreSQL ni de R2."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
from io import StringIO
from itertools import islice
from pathlib import Path
import re

import numpy as np
import pandas as pd


CSV_COLUMNS = {
    "P": "pressure_hPa", "Height": "height_m", "T": "temp_C",
    "TD": "dewpoint_C", "FF": "wind_speed_ms", "DD": "wind_dir_deg",
    "RH": "rh_percent", "MR": "mixing_ratio_gkg",
}
EDT_UNITS = {
    "P": "hPa", "Height": "m", "T": "K", "TD": "K", "FF": "m/s",
    "DD": "dgr", "RH": "%", "MR": "g/kg",
}


def read_metadata(path):
    """Usa Launch time del encabezado; los nombres de archivo no definen la fecha."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        launch = datetime.strptime(path.stem, "%Y-%m-%d-%HZ").replace(tzinfo=timezone.utc)
        return {"path": path, "launch_time": launch, "cycle": launch, "station": None}
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        header = "".join(islice(source, 100))
    match = re.search(
        r"^Launch time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\s*$",
        header, re.MULTILINE,
    )
    if not match:
        raise ValueError(f"{path.name}: falta Launch time UTC válido.")
    launch = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    midnight = launch.replace(hour=0, minute=0, second=0)
    candidates = [midnight + timedelta(hours=hour) for hour in (0, 12, 24)]
    cycle = min(candidates, key=lambda value: abs((value - launch).total_seconds()))
    if abs((cycle - launch).total_seconds()) > 3600:
        cycle = None
    station = re.search(r"^Station:\s*(.+?)\s*$", header, re.MULTILINE)
    return {
        "path": path, "launch_time": launch, "cycle": cycle,
        "station": station.group(1) if station else None,
    }


def build_catalog(folder):
    """Indexa ciclos UTC de 12 h y deduplica sólo archivos idénticos."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"No existe la carpeta local de radiosondeos: {folder}")
    records, invalid, outside = [], [], []
    grouped = defaultdict(list)
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tsv", ".csv"}:
            continue
        try:
            record = read_metadata(path)
        except ValueError as exc:
            invalid.append({"file": path.name, "error": str(exc)})
            continue
        records.append(record)
        if record["cycle"] is None:
            outside.append(record)
        else:
            grouped[record["cycle"]].append(record)
    catalog, duplicates, conflicts = {}, [], []
    for cycle, group in sorted(grouped.items()):
        if len(group) > 1:
            hashes = {hashlib.sha256(record["path"].read_bytes()).hexdigest() for record in group}
            detail = {"cycle": cycle.isoformat(), "files": [record["path"].name for record in group]}
            if len(hashes) > 1:
                conflicts.append(detail)
                continue
            duplicates.append(detail)
        catalog[cycle] = group[0]
    return {
        "records": records, "cycles": catalog, "invalid": invalid,
        "outside_cycles": outside, "identical_duplicates": duplicates,
        "conflicting_cycles": conflicts,
    }


def select_window(catalog, start, size):
    """Exige todos los ciclos; no sustituye faltantes por otros días."""
    expected = [start + timedelta(hours=12 * index) for index in range(size)]
    missing = [cycle for cycle in expected if cycle not in catalog["cycles"]]
    if missing:
        return None, {
            "error": "No hay una ventana completa de radiosondeos separados por 12 horas.",
            "ciclos_faltantes": [cycle.strftime("%Y-%m-%dT%H:%M:%SZ") for cycle in missing],
            "ciclos_ambiguos": catalog["conflicting_cycles"],
        }
    records = [catalog["cycles"][cycle] for cycle in expected]
    stations = {record["station"] for record in records if record["station"]}
    if len(stations) > 1:
        return None, {"error": "La ventana contiene radiosondeos de estaciones diferentes."}
    return [record["path"] for record in records], None


def read_sounding(path):
    """Convierte EDT a las columnas y unidades del modelo, sin guardar CSV."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()
        header_index = next(
            (index for index, line in enumerate(lines) if {"time", "P", "T", "TD", "Height"}.issubset(line.split())),
            None,
        )
        if header_index is None:
            raise ValueError(f"{path.name}: no hay encabezado EDT reconocible.")
        header = "\n".join(lines[:header_index])
        for column, expected_unit in EDT_UNITS.items():
            match = re.search(rf"^\s*{column}\s+(\S+)\s+", header, re.MULTILINE)
            if not match or match.group(1) != expected_unit:
                raise ValueError(f"{path.name}: unidad no compatible para {column}; se esperaba {expected_unit}.")
        raw = pd.read_csv(StringIO(text), sep=r"\s+", skiprows=header_index, engine="python")
        missing = set(CSV_COLUMNS).difference(raw.columns)
        if missing:
            raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}.")
        dataframe = raw[list(CSV_COLUMNS)].rename(columns=CSV_COLUMNS)
        dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
        dataframe = dataframe.replace([-32768.0, np.inf, -np.inf], np.nan)
        dataframe["temp_C"] -= 273.15
        dataframe["dewpoint_C"] -= 273.15
    missing = set(CSV_COLUMNS.values()).difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas del modelo {sorted(missing)}.")
    dataframe = dataframe[list(CSV_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    dataframe = dataframe.replace([-32768.0, np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna(subset=["height_m"])
    dataframe = dataframe[dataframe["height_m"] >= 0]
    if len(dataframe) < 2:
        raise ValueError(f"{path.name}: no hay alturas suficientes para interpolar.")
    # Limitar al ascenso hasta el primer tope y ordenar alturas repetidas.
    top = int(np.argmax(dataframe["height_m"].to_numpy()))
    dataframe = dataframe.iloc[:top + 1]
    return dataframe.groupby("height_m", as_index=False).mean().sort_values("height_m").reset_index(drop=True)
