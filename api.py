import requests
import datetime

API_URL = "https://api-inference.huggingface.co/models/Meina/MeinaMix_V10"
API_TOKEN = open("huggingface_access_token.txt", "r").read().rstrip("\n")

print(API_TOKEN)

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

image_bytes = query({
	"inputs": "Astronaut riding a horse",
})

print(str(image_bytes))
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image_path = "./" + str(datetime.datetime.now().strftime("%H%M%S%f")) + ".png"
image.save("./{image_path}.png")
