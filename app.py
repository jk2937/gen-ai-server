# File: app.py
from flask import *
import threading
import time
import datetime

import genai

genai.init()

app = Flask(__name__)

save_folder = './files/img/'

# Simulate generated images with base64 encoded strings
images = []
generating = False
generated_count = 0
total_to_generate = 0

def generate_images(prompt, resolution, num_images):
    global generating, generated_count, total_to_generate
    for _ in range(num_images):
        image_path = save_folder + str(datetime.datetime.now().strftime("%H%M%S%f")) + ".png"
        genai.startGeneration(prompt, int(252 * resolution), int(448 * resolution), image_path)
        images.append(image_path)
        generated_count += 1
    generating = False

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory('files', filename)

@app.route('/start', methods=['POST'])
def start():
    global generating, generated_count, total_to_generate
    data = request.json
    prompt = data.get('prompt')
    resolution = int(data.get('resolution'))
    num_images = int(data.get('num_images'))
    total_to_generate = num_images
    generating = True
    threading.Thread(target=generate_images, args=(prompt, resolution, num_images)).start()
    return jsonify(success=True)

@app.route('/status')
def status():
    return jsonify(generating=generating, generated_count=generated_count, total_to_generate=total_to_generate, images=images)

if __name__ == '__main__':
    app.run(debug=True)

