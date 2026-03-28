import requests

url = "http://127.0.0.1:5000/predict"

# dummy transaction (29 features)
data = {
    "features": [0]*29
}

response = requests.post(url, json=data)

print(response.json())