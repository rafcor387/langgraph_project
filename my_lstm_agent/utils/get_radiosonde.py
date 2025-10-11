from config.mongodb import collection, client
from utils.calculations import calculos

def get_radiosonde_fromDB(date: str):
    doc = collection.find_one({"date": date})

    if doc:
        # Capturar los datos en variables
        radiosonde_data = {
            "launch_time": doc.get("launch_time"),
            "date": doc.get("date"),
            "time": doc.get("time"),
            "pressure": doc.get("pressure", []),
            "height": doc.get("height", []),
            "temperature": doc.get("temperature", []),
            "dewpoint": doc.get("dewpoint", []),
            "wind_speed": doc.get("wind_speed", []),
            "wind_dir": doc.get("wind_dir", [])
        }

        #radiosonde_calculations = calculos(radiosonde_data)

        return radiosonde_data
    else:
        return None
    