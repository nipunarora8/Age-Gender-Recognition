from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os,cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import *
app = Flask(__name__)

gender_model=load_model("gender_new.h5")
age_model=load_model("age_best.h5")

age_classes=['0-5', '12-17', '18-30', '30-50', '50+', '6-11']

def model_predict(img_path, age_model,gender_model):
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img,1.3,5)
    x,y,w,h = faces[0]
    imgg=img[y:y+h,x:x+h]
    imgg=cv2.resize(imgg,(64,64))/255.0
    imgg=imgg.reshape(1,64,64,3)

    pred_gender=gender_model.predict(imgg)
    pred_age=age_model.predict_classes(imgg)
    pred_a=age_classes[pred_age[0]]
    
    if pred_gender[0][0]>0.5:
        pred_g="Male"
    else:
        pred_g="Female"

    return (pred_a,pred_g)



@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        age,gender=model_predict(file_path,age_model,gender_model)

        return render_template('pred.html',age=age,gender=gender,file_name=str(f.filename))
	
@app.route('/upload/<filename>')
def upload_img(filename):
    return send_from_directory("uploads", filename)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)