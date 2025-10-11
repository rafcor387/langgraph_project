from langchain_core.tools import tool
from utils.get_radiosonde import get_radiosonde_fromDB
from utils.calculations import calculos

@tool
def get_radiosonde(date: str) -> str:
    """Find a radiosonde from the database
    if the radiosende was found return the next arrays: 
        Pressure, temperature, dewpoint, wind_speed, wind_direction, height;
        It also return date, launch_time, time.
    else return not found

    Args:
        date: date in year-month-day YYYY-MM-DD format e.g 2018-12-29
    """
    radiosonde = get_radiosonde_fromDB(date)

    if radiosonde:
        data = calculos(radiosonde)
        return f"the data get to the date {date}: {data}"
    
    return f" the radiosonde with date {date} was not found"

@tool
def find_number_by_name(name: str) -> int:
    """Finds a phone number by name.
    Args:
        name: The name of the person to find the phone number for."""
    
    phone_book = {
        "Alice": "719-65390",
        "Roberto": "656-15108",
        "Manuel": "60595837",
        "Sofia": "71984355"
    }
    return phone_book.get(name, f"Sorry, I couldn't find a number for {name}.")

@tool
def delete_number_by_name(name: str) -> str:
    """Deletes a phone number by name.
    Args:
        name: The name of the person to delete the phone number for."""
    
    if name in ["Alice", "Roberto"]:
        return f"The contact for '{name}' has been successfully deleted."
    else:
        return f"Could not delete '{name}' as they were not found in the contact list."
    
@tool
def travel_outspace(destination: str, discount: int = 0) -> str:
    """Simulates traveling to a destination in outer space.
        It has a list of known destinations: Mars, Moon, Jupiter, Saturn.
        It's optiocal aplicate a discount code to lower the price of the ticket.
    Args:
        destination: The name of the destination to travel to.
        discount: (Optional) The discount percentage to apply to the ticket price."""
    
    destinations_Ticketprice = {
        "Mars": 500,
        "Jupiter": 250.5,
        "Moon": 150,
        "Saturn" : 900
    }

    if discount > 0:
        destinations_Ticketprice = {
        "Mars": 500 - (discount * (500/100)),
        "Jupiter": 250.5 - (discount * (250.5/100)),
        "Moon": 150 - (discount * (150/100)),
        "Saturn" : 900 - (discount * (900/100))
        }
        return destinations_Ticketprice.get(destination, f"There is a discount but the destination does not exist")
    
    else:
        return destinations_Ticketprice.get(destination, f"There is not discount and the destination does not exist")

@tool
def check_discount(code: str) -> str:
    """Checks if a discount code is valid and returns the discount percentage.
    Args:
        code: The discount code to check."""
    
    valid_codes = {
        "SAVE10": 2,
        "SAVE20": 4,
        "SAVE30": 7
    }
    return valid_codes.get(code, f"The discount code {code} was invalid")

tools = [find_number_by_name, delete_number_by_name, travel_outspace, check_discount, get_radiosonde]