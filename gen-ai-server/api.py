import requests

API_URL = "https://api-inference.huggingface.co/models/Meina/MeinaMix_V10"
API_TOKEN = open("huggingface_access_token.txt", "r").read()

print(API_TOKEN)

headers = {"Authorization": f"Bearer {API_TOKEN}"}

print(headers)


quit()

print("foobar")


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
