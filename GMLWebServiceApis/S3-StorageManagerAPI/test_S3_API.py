import requests
import os

# Generate a random file for the demo
filename = "test.txt"
# with open(filename, "w") as f:
#     f.write(os.urandom(1024).hex())  # Writes 1KB of random data

# Define the API endpoint and the headers
url = "http://206.12.99.253:8443/model"
headers = {"accept": "application/json"}

# Perform the file upload
with open(filename, "rb") as f:
    response = requests.post(url, files={"model_file": f}, headers=headers)

# Print the response from the server
print(response.status_code)
print(response.text)

# Use the response to get the model's path in S3
response_data = response.json()
s3_path = response_data.get("s3_path")

# Fetch the model from the server
get_url = f"http://206.12.99.253:8443/model/{filename}"
response_get = requests.get(get_url, headers=headers)

# Print the status and save the retrieved file for verification if needed
print(response_get.status_code)
# Print the content of the retrieved file
print("Content of the retrieved file:")
print(response_get.text)

# Optionally remove the random file if you no longer need it
# os.remove(filename)