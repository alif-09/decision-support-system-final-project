import requests

url = 'http://127.0.0.1:8080/predict'
data = {"text": "i take the rope, and i want to hang on it"}
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed with status code:", response.status_code)
