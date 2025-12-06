from langchain_core.tools import tool
from utils.get_radiosonde import get_radiosonde_fromDB
from utils.calculations import calculos
from tools.weather_tools import classify_weather_pattern
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


tools = [get_radiosonde_from_dataset,classify_weather_pattern,diagram_skew_t]
