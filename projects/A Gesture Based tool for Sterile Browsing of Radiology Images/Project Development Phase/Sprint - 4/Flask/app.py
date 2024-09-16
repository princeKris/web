from flask import Flask,render_template,request
import operator
import cv2 # opencv library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder="templates") # initializing a flask app
# Loading the model
model=load_model('gesture.h5')
print("Loaded model from disk")


@app.route('/')# route to display the home page
def home():
    return render_template('home.html')#rendering the home page

@app.route('/intro') # routes to the intro page
def intro():
    return render_template('intro.html')#rendering the intro page

@app.route('/image1',methods=['GET','POST'])# routes to the index html
def image1():
    return render_template("launch.html")

@app.route('/predict',methods=['GET', 'POST'])# route to show the predictions in a web UI
def launch():
    if request.method == 'POST':
        print("inside image")
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)   
        print(file_path)
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read() #capturing the video frame values
            frame = cv2.flip(frame, 1)# Simulating mirror image
            x1 = int(0.5*frame.shape[1]) # Coordinates of the ROI
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)# Drawing the ROI
            roi = frame[y1:y2, x1:x2]# Extracting the ROI
            # Resizing the ROI so it can be fed to the model for prediction
            roi = cv2.resize(roi, (64, 64)) 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
            cv2.imshow("test", test_image)
            # Batch of 1
            result = model.predict(test_image.reshape(1, 64, 64, 1))
            prediction = {'ZERO': result[0][0], 
                          'ONE': result[0][1], 
                          'TWO': result[0][2],
                          'THREE': result[0][3],
                          'FOUR': result[0][4],
                          'FIVE': result[0][5]}
            # Sorting based on top prediction
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            # Displaying the predictions
            cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
            cv2.imshow("Frame", frame)
            
            #loading an image
            image1=cv2.imread(file_path)
            if prediction[0][0]=='ZERO':
                cv2.waitKey(0)
            
            
            elif prediction[0][0]=='ONE':
                try:
                    cv2.destroyWindow("op")
                except:
                    print("1")
                finally:
                    resized = cv2.resize(image1, (200, 200))
                    cv2.imshow("op", resized)
                    #cv2.destroyWindow("op")
                    #cv2.rectangle(image1, (480, 170), (650, 420), (0, 0, 255), 2)
                    #cv2.imshow("op", image1)
                
                
            elif prediction[0][0]=='THREE':
                try:
                    cv2.destroyWindow("op")
                except:
                    print("3")
                finally:
                    (h, w, d) = image1.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -45, 1.0)
                    rotated = cv2.warpAffine(image1, M, (w, h))
                    cv2.imshow("op", rotated)
                
            elif prediction[0][0]=='TWO':
                try:
                    cv2.destroyWindow("op")
                except:
                    print("2")
                finally:
                    blurred = cv2.GaussianBlur(image1, (21, 21), 0)
                    cv2.imshow("op", blurred)

            elif prediction[0][0]=='FOUR':
                try:
                    cv2.destroyWindow("op")
                except:
                    print("4")
                finally:
                    resized = cv2.resize(image1, (400, 400))
                    cv2.imshow("op", resized)

            elif prediction[0][0]=='FIVE':
                try:
                    cv2.destroyWindow("op")
                except:
                    print("5")
                finally:
                    gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
                    cv2.imshow("op", gray)

            else:
                continue
            
            
            interrupt = cv2.waitKey(10)
            if interrupt & 0xFF == 27: # esc key
                break
                
         
        cap.release()
        cv2.destroyAllWindows()
    return render_template("home.html")
     
if __name__ == "__main__":
   # running the app
    app.run(debug=True)
