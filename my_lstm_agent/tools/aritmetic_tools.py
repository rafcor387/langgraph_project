from langchain_core.tools import tool

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
def travel_outspace(destination: str) -> str:
    """Simulates traveling to a destination in outer space.
        It has a list of known destinations: Mars, Moon, Jupiter, Saturn.
    Args:
        destination: The name of the destination to travel to."""
    
    known_destinations = ["Mars", "Moon", "Jupiter", "Saturn"]
    if destination in known_destinations:
        return f"Ticket bought for {destination}!"
    else:
        return f"Destination '{destination}' is unknown. Please choose a valid space destination."
    
@tool
def check_discount(code: str) -> str:
    """Checks if a discount code is valid and returns the discount percentage.
    Args:
        code: The discount code to check."""
    
    valid_codes = {
        "SAVE10": 10,
        "SAVE20": 20,
        "SAVE30": 30
    }
    discount = valid_codes.get(code)
    if discount:
        return f"Discount code '{code}' is valid for {discount}% off."
    else:
        return f"Discount code '{code}' is invalid."

tools = [find_number_by_name, delete_number_by_name, travel_outspace, check_discount]