from flask import Flask, request,jsonify
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Load the Custom trained Model
model = YOLO("goi_v2.pt")


# Post API to recieve image and email, and save them
@app.route('/api/predict', methods=['POST'])
def handle_image_input():
    if 'image_input' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    email = request.form.get('email')
    if not email:
        return "Please provide a valid email"
    
    image_file = request.files['image_input']
    image_file.save(os.path.join('calls/', image_file.filename))
    # Open the image using Pillow
    image = Image.open(image_file)
    print("Image received successfully")
    
    response = {
        'email': email,
        'message': 'Image received successfully'
    }
    
    with open('response.json', 'w') as f:
        json.dump(response, f)
    return jsonify(response)

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    return jsonify(predictions)


# Post API to predict objects from saved image

@app.route('/api/process_image', methods=['POST'])
def process_image():
    with open('response.json', 'r') as f:
        response = json.load(f)

    if response['message'] != 'Image received successfully':
        return "Please provide a valid image"

    directory = 'calls'

    # Get all files in the directory
    files = os.listdir(directory)

    # Filter out non-image files (optional)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    # Sort the files by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # Get the path to the newest image file
    newest_image_path = os.path.join(directory, image_files[0])
    # Process the image with your YOLO model and return the predictions
    
    image = Image.open(newest_image_path)
    predictions = model(image, imgsz=640)[0]
    object_dct = {'Apple': 0, 'Banana': 0, 'Bread': 0, 'Carrot': 0, 'Tomato': 0, 'Potato': 0, 'Orange': 0}
    objects = []

    for r in predictions:
        for c in r.boxes.cls:
            objects.append(model.names[int(c)])
            print(model.names[int(c)])

    for item in objects:
        if item in object_dct:
            object_dct[item] += 1
        else:
            object_dct[item] = 1

    response['predictions'] = object_dct
    
    with open('predictions.json', 'w') as f:
        json.dump(response, f)
    return jsonify(response)



if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #app.run()