import numpy as np
import random


def getage(prediction,classnames):
    agemap = {
        "clear skin": (18, 30),
        "dark spots": (30, 45),
        "puffy eyes": (45, 60),
        "wrinkles": (60, 100)
    }

    probs = prediction[0]
    maximum = np.argmax(probs)
    dominantclass = classnames[maximum]

    low, high = agemap[dominantclass]

    return random.randint(low, high)

def predict(img,model,faces):
    
    classnames = ["clear skin","dark spots","puffy eyes","wrinkles"]

    prediction = model.predict(img,verbose=0)
    print(prediction)
    predictedindex = np.argmax(prediction)
    confidence = prediction[0][predictedindex] * 100
    label = classnames[predictedindex]

    return {
        "label": label,
        "confidence": confidence,
        "age": getage(prediction,classnames)
    }

    #cv2.imshow("Face Detection & Prediction", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
