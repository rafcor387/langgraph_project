

def calculos(radiosonde: dict) -> dict:

    launch_time = radiosonde['launch_time']
    data={
        "launch" : launch_time,
        "CAPE": 1205,
        "CIn":200,
        "lapse rate enviroment":"4 K/Km"
    }

    return data