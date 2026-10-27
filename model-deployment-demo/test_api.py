import requests

# Input data with 30 features (ensure these are meaningful values)
data = {
    "features": [
        15.0, 20.25, 98.5, 0.08, 0.047, 0.01, 0.2, 0.14, 0.32, 0.07, 0.24, 0.06, 0.06, 0.18,
        0.09, 0.09, 0.1, 0.23, 0.19, 0.1, 0.07, 0.27, 0.42, 0.05, 0.17, 0.06, 0.23, 0.3,
        0.05, 0.06 # 30 features total
    ]
}

# Send the POST request to FastAPI
response = requests.post("http://127.0.0.1:8000/predict/", json=data)

# Print the response (prediction)
<<<<<<< HEAD
print(response.json())
=======
print(response.json()) 
>>>>>>> d91ba9f6edf6b0e9d6d7403b9636ca9129242bfb
