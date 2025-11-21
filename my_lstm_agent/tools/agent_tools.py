from langchain_core.tools import tool
from utils.get_radiosonde import get_radiosonde_fromDB
from utils.calculations import calculos
import pandas as pd
import json    


@tool
def get_radiosonde(date: str) -> str:
    """Find a radiosonde from the database
    if the radiosende was found return the next values:
    date, CAPE Superficie, CIn Superficie
        It also return date, launch_time, time.
    else return not found

    Args:
        date: date in year-month-day YYYY-MM-DD format e.g 2018-12-29
    Return:
        the return It must contain, in addition to the radiosonde data, 
        its respective interpretation of the atmosphere based on the data obtained.
    """
    radiosonde = get_radiosonde_fromDB(date)

    if radiosonde:
        data = calculos(radiosonde)
        return f"the data obtain by date {date}: {data}"
    
    return f" the radiosonde with date {date} was not found"

@tool
def get_radiosonde_from_dataset(date: str):
    """Find a radiosonde from the dataset
    if the radiosende was found return the next array: 
        Pressure, temperature, dewpoint, wind_speed, wind_direction, height;
        It also return date, launch_time, time.
    else return not found

    Args:
        date: date in year-month-day YYYY-MM-DD format e.g 2018-12-29
    """
    df = pd.read_csv("../my_lstm_agent/dataset/labels.csv", parse_dates=["date"])

    row = df.loc[df["date"] == date]

    if row.empty:
        return("No se encontró ningún registro con esa fecha.")
    else:
        # Convertir la primera fila al diccionario con esas columnas
        data_full = row.iloc[0].to_dict()

        data = json.dumps(data_full, indent=2)
        # Imprimir bonito (opcional)
        return f"at the end of your response, must ask if the user wants the data from radiosonde date 2018-12-01",data

import duckdb, pandas as pd, json
 
@tool("duckdb_sql")
def duckdb_sql(sql: str, csv_path: str = "../my_lstm_agent/dataset/data.csv") -> str:
    """
    Ejecuta SQL sobre el CSV usando DuckDB.
    Ejemplos:
      SELECT BankUserFullName, SUM(Monto) AS total
      FROM tabla
      WHERE lower(BankUserFullName) LIKE '%marce%'
      GROUP BY 1
      ORDER BY total DESC
      LIMIT 10
    """
    # Cargar CSV a un DataFrame (o registrar directamente el archivo en DuckDB)
    df = pd.read_csv("../my_lstm_agent/dataset/data.csv")
 
    # Registrar DataFrame como tabla 'tabla'
    con = duckdb.connect()
    con.register("tabla", df)
 
    # Ejecutar SQL y devolver JSON
    res = con.execute(sql).df()
    out = {
        "ok": True,
        "columns": list(res.columns),
        "rows": res.to_dict(orient="records"),
        "row_count": int(len(res))
    }
    return json.dumps(out, ensure_ascii=False)

tools = [get_radiosonde_from_dataset,duckdb_sql]
