import cv2,os,numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

scriptspath = os.path.dirname(os.path.abspath(__file__))

facecascade = cv2.CascadeClassifier(os.path.join(scriptspath,"haarcascade_frontalface_default.xml"))

img = cv2.imread(r"C:\Users\adith\OneDrive\Pictures\Camera Roll\WIN_20251221_09_36_52_Pro.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

print(faces)



if(len(faces)>0):
    x,y,w,h = faces[0]
    faceimg = img[y:y+h,x:x+w]
else:
    faceimg = img.copy()


#cv2.imshow("Cropped Face", faceimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

faceimg = cv2.resize(faceimg,(224,224))
faceimg = preprocess_input(faceimg)
faceimg = np.expand_dims(faceimg,axis=0)

print(faceimg.shape)

classnames = ["clear skin","dark spots","puffy eyes","wrinkles"]

model = tf.keras.models.load_model("models/resnet_best_model.h5")
prediction = model.predict(faceimg)
print(prediction)
predictedindex = prediction.argmax()
confidence = prediction[0][predictedindex] * 100
label = classnames[predictedindex]

agemapping = {
    "clear skin": "18-25",
    "dark spots": "25-35",
    "puffy eyes": "30-45",
    "wrinkles": "40+"
}



if(len(faces)>0):
    cv2.putText(
        img,
        f"{label}: {confidence:.2f}% | Age: {agemapping[label]}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )
else:
    cv2.putText(
        img,
        f"{label}: {confidence:.2f}% | Age: {agemapping[label]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


cv2.imshow("Face Detection & Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
