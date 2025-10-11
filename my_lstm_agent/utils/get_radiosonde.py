from config.mongodb import collection, client

def get_radiosonde_bydate(date: str):
    # Buscar el documento por su _id
    doc = collection.find_one({"date": date})

    if doc:
        # Capturar los datos en variables
        launch_time = doc.get("launch_time")

        return launch_time
    else:
        return None
    