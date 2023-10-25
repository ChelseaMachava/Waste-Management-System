from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("my_model1.h5")
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.pdf']

def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((150, 150))
    # Convert image to array and normalize pixel values
    image_array = img_to_array(image) / 255.0
    # Expand dimensions to match the model's input shape
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

@app.route("/")
def hello():
    return render_template('index.html', prediction_text='')

@app.route("/predict", methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Open the uploaded file as an image
        image = Image.open(uploaded_file)
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make the prediction using the model
        prediction = model.predict(processed_image)
        # Get the predicted class label
        predicted_class = np.argmax(prediction)

        # Perform any further processing on the predicted class if needed

        prediction_text = f"This type of waste is {predicted_class}"
        return render_template('index.html', prediction_text=prediction_text)

    return render_template('index.html', prediction_text='No file selected.')

if __name__ == "__main__":
    app.run()
