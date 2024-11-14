# file: send_request.py
import requests
import json

# Define the API URL
url = "http://localhost:8000/predict"

# Load the example input JSON file
with open("example_input.json", "r") as file:
    input_data = json.load(file)

try:
    # Send a POST request to the API
    response = requests.post(url, json=input_data)
    print("Response status:", response.status_code)
    response.raise_for_status()  # Check for HTTP errors
    # Print the prediction response
    print("Prediction:", response.json())
except requests.exceptions.RequestException as e:
    print("Error during API request:", e)