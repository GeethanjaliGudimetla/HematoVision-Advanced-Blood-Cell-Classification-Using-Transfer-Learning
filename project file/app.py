from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64

app = Flask(__name__)
model = load_model("model_training/Blood_Cell.h5")
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (244, 244))
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
    predictions = model.predict(img_preprocessed)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class, img_rgba

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("home.html", error="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("home.html", error="No selected file")
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        prediction, image = predict_image_class(filepath)
        _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode()
        return render_template("result.html", prediction=prediction, image_data=img_str)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
