import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import interpolate_1d, log_interpolate_1d
import metpy.constants as mpconst
import pandas as pd
import numpy as np

def calculos(radiosonde: dict) -> dict:
    # ========= VARIABLES PRINCIPALES =========
    pressure = np.array(radiosonde['pressure']) * units.hPa
    temperature = np.array(radiosonde['temperature']) * units.kelvin
    dewpoint = np.array(radiosonde['dewpoint']) * units.kelvin
    height = np.array(radiosonde['height']) * units.meter
    wind_speed = np.array(radiosonde['wind_speed']) * units.meter / units.second
    wind_dir = np.array(radiosonde['wind_dir']) * units.degrees

    # ========= NIVELES CARACTERÍSTICOS Y PARCELAS =========
    p_sfc, T_sfc, Td_sfc = pressure[0], temperature[0], dewpoint[0]
    prof_sb = mpcalc.parcel_profile(pressure, T_sfc, Td_sfc)
    lcl_p, lcl_T = mpcalc.lcl(p_sfc, T_sfc, Td_sfc)

    launch_time = radiosonde['launch_time']
    data={
        "launch" : launch_time,
        "Nivel de condensación por elevación (LCL) presión": lcl_p,
        "Nivel de condensación por elevación (LCL) temperatura":lcl_T,
        "Presión superficie": p_sfc,
        "Temperatura superficie": T_sfc,
        "Temperatura de rocío superficie": Td_sfc
    }

    if data:
        return data
    return "Error try to generate radiosonde"