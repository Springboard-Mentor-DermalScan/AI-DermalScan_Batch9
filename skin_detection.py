#Importing necessary libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Model and file paths mentioned
model_path = "MobileNetV2_best1_notebook2.h5"
face_cascade_path = "haarcascade_frontalface_default.xml"
test_folder = "Dataset_split/test"
img_size = 224
conf_threshold = 50.0

output_folder = "outputs"

correct_dir = os.path.join(output_folder, "correct_predictions")
wrong_dir = os.path.join(output_folder, "wrong_predictions")
uncertain_dir = os.path.join(output_folder, "uncertain_predictions")

os.makedirs(correct_dir, exist_ok=True)
os.makedirs(wrong_dir, exist_ok=True)
os.makedirs(uncertain_dir, exist_ok=True)

#Loading Model and Face Detector
model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)


class_labels = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]
age_map = {
    "Clear Skin": "18–30",
    "Dark Spots": "25–45",
    "Puffy Eyes": "20–40",
    "Wrinkles": "40+"
}

#Image Preprocessing Function
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (img_size, img_size))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)


#Defining counters for statistics
total_images = 0
correct = 0
wrong = 0
uncertain = 0
no_face_detected = 0


#Processing each image in the test dataset (15% of total data)
for true_class in os.listdir(test_folder):
    class_path = os.path.join(test_folder, true_class)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        total_images += 1
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
        )

        print("\n==============================")
        print("Image Name :", img_name)
        print("Actual     :", true_class)

        if len(faces) == 0:
            print("⚠ No face detected")
            no_face_detected += 1
            continue

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            face = image[y:y+h, x:x+w]
            preds = model.predict(preprocess_face(face), verbose=0)[0]

            best_idx = np.argmax(preds)
            confidence = preds[best_idx] * 100

            if confidence < conf_threshold:
                label = "Uncertain"
                status = "UNCERTAIN"
                uncertain += 1
            else:
                label = class_labels[best_idx]
                age = age_map[label]
                normalized_pred = label.lower().replace(" ", "_")
                normalized_true = true_class.lower().replace(" ", "_")
                if normalized_pred == normalized_true:
                    status = "CORRECT"
                    correct += 1
                else:
                    status = "WRONG"
                    wrong += 1
                  
            print("Predicted  :", label)
            print("Age Group  :", age)
            print(f"Confidence : {confidence:.2f}%")
            print("Status     :", status)

            # Draw box
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

            # Display text on image
            text = f"{label} ({age}) | {confidence:.1f}% | {status}"
            if status == "CORRECT":
                color = (0, 255, 0)     
            elif status == "WRONG":
                color = (0, 0, 255)      
            else:
                color = (0, 255, 255)    

            cv2.putText(
                image, text, (x, max(y-10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2
            )
            
            #Save Images: folders classified as correct, wrong and uncertain
            if status == "CORRECT":
                save_dir = correct_dir
            elif status == "WRONG":
                save_dir = wrong_dir
            else:
                save_dir = uncertain_dir

            base_name = os.path.splitext(img_name)[0]
            save_name = f"{base_name}_face{idx}.jpg"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, image)


        # Show image
        display = cv2.resize(image, None, fx=1.2, fy=1.2)
        cv2.imshow("DermalScan - Test Evaluation", display)
        cv2.waitKey(0)

cv2.destroyAllWindows()


#Calculating final accuracy and printing the test summary
accuracy = correct / max((correct + wrong), 1)

print("\n===================================")
print("FINAL EVALUATION SUMMARY")
print("===================================")
print("Total Test Images        :", total_images)
print("Correct Predictions      :", correct)
print("Wrong Predictions        :", wrong)
print("Low-Confidence (Uncertain):", uncertain)
print("No Face Detected         :", no_face_detected)
print("\nNote: The face not detected is because of the detector's limitation.\n")
print(f"Final Model Accuracy: {accuracy*100:.2f}%")


#Plotting the results - Number of correct, wrong and uncertain predictions  

plt.figure(figsize=(6,4))
plt.bar(
    ["Correct", "Wrong", "Uncertain", "No Face Detected"],
    [correct, wrong, uncertain, no_face_detected],
    color=["green", "red", "orange", "black"]
)

plt.title("DermalScan Prediction Summary")
plt.ylabel("Number of Images")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
