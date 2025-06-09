### Getting the config data part ###
# Function to recursively convert a dictionary to a SimpleNamespace object (We use it below for getting the config file)
import json
from types import SimpleNamespace
from typing import Dict

from configModels import Config


def convert_to_namespace(data):
    if isinstance(data, dict):
        # Create a new dictionary where each value is recursively converted
        converted_dict: Dict = {}
        for key, value in data.items():
            # Recursively convert the value (could be another dict, list, or primitive)
            converted_dict[key] = convert_to_namespace(value)
            
        return SimpleNamespace(**converted_dict)
    
    # If the input is not a dictionary, return it unchanged
    return data

# Load the configuration from the JSON file
try:
    with open('pythonService/config.json', 'r') as config_file:
        # Parse the JSON file into a Python dictionary
        config_dict = json.load(config_file)
    
    # Convert the dictionary to a SimpleNamespace object for dot notation
    config: Config = convert_to_namespace(config_dict)

except FileNotFoundError:
    print("Error: The file 'config.json' was not found.")
    raise
except json.JSONDecodeError:
    print("Error: The 'config.json' file contains invalid JSON.")
    raise