import cv2
import os
from werkzeug.utils import secure_filename
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

model = load_model('alex1.h5')

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(path):
    ref = {0: 'Bacterial_spot',1: 'Early_blight',2: 'healthy',3: 'Late_blight',4: 'Leaf_Mold',5: 'Mosaic_virus',6: 'Septoria_leaf_spot',7: 'Spider_mite',8: 'Target_spot',9: 'Yellow_Leaf_curl_Virus'}
    img = load_img(path,target_size=(128,128))
    # img = img_to_array(img)
    # img = preprocess_input(img)
    # img = np.expand_dims(img,axis=0)
    img = np.array(img).reshape(-1,128,128,3)
    print(model.predict(img))
    pred = np.argmax(model.predict(img))
    return ref[pred]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pred = prediction(UPLOAD_FOLDER+'/'+filename)
        return render_template('home.html',org_img_name=filename, prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
