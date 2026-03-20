import numpy as np
import os
import cv2
import csv


args = {
    "face": r"D:\WMiT_Laboratory\projekty\analiza obrazu detekcja wieku i humoru\face_detector",
    "age": r"D:\WMiT_Laboratory\projekty\analiza obrazu detekcja wieku i humoru\age_detector",
    "gender": r"D:\WMiT_Laboratory\projekty\analiza obrazu detekcja wieku i humoru\gender_detector",
    "confidence": 0.3
}

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

GENDER_BUCKETS = ["Male", "Female"]

AGE_MIDPOINTS = {
    "(0-2)": 1.0,
    "(4-6)": 5.0,
    "(8-12)": 10.0,
    "(15-20)": 17.5,
    "(25-32)": 28.5,
    "(38-43)": 40.5,
    "(48-53)": 50.5,
    "(60-100)": 80.0
}

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading gender detector model...")
prototxtPath = os.path.sep.join([args["gender"], "gender_deploy.prototxt"])
weightsPath = os.path.sep.join([args["gender"], "gender_net.caffemodel"])
genderNet = cv2.dnn.readNet(prototxtPath, weightsPath)


imageDir = r"D:\WMiT_Laboratory\projekty\analiza obrazu detekcja wieku i humoru\pictures"
outputDir = r"D:\WMiT_Laboratory\projekty\analiza obrazu detekcja wieku i humoru\output"

os.makedirs(outputDir, exist_ok=True)

peopleCsvPath = os.path.join(outputDir, "wyniki_osoby.csv")
summaryCsvPath = os.path.join(outputDir, "podsumowanie.csv")

with open(peopleCsvPath, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "image_name",
        "person_id",
        "age_range",
        "age_confidence",
        "estimated_age",
        "gender",
        "gender_confidence"
    ])

all_estimated_ages=[]
female_count = 0
male_count = 0
total_people = 0

files = [f for f in os.listdir(imageDir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def sort_key(filename):
    name, ext = os.path.splitext(filename)
    return (0, int(name)) if name.isdigit() else (1, name.lower())

files.sort(key=sort_key)

for file in files:
    imagePath = os.path.join(imageDir, file)
    print("[INFO] processing image...", imagePath)

    image = cv2.imread(imagePath)
    if image is None:
        continue

    image = cv2.resize(src=image, dsize=None, fx=1.2, fy=1.2)

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()
    person_id = 0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = image[startY:endY, startX:endX]

            if face.size == 0:
                continue

            person_id += 1
            total_people += 1

            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            ageNet.setInput(faceBlob)
            agePreds = ageNet.forward()
            ageIdx = agePreds[0].argmax()
            age = AGE_BUCKETS[ageIdx]
            ageConfidence = float(agePreds[0][ageIdx])
            estimatedAge = AGE_MIDPOINTS[age]

            genderNet.setInput(faceBlob)
            genderPreds = genderNet.forward()
            genderIdx = genderPreds[0].argmax()
            gender = GENDER_BUCKETS[genderIdx]
            genderConfidence = float(genderPreds[0][genderIdx])

            all_estimated_ages.append(estimatedAge)

            if gender == "Female":
                female_count += 1
            else:
                male_count += 1

            text = f"ID {person_id} | {gender}: {genderConfidence * 100:.2f}% | {age}: {ageConfidence * 100:.2f}%"
            print(f"[INFO] {file} -> {text}")

            with open(peopleCsvPath, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    file,
                    person_id,
                    age,
                    round(ageConfidence, 4),
                    estimatedAge,
                    gender,
                    round(genderConfidence, 4)
                ])


            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    outputPath = os.path.join(outputDir, file)
    cv2.imwrite(outputPath, image)

if len(all_estimated_ages) > 0:
    mean_age = sum(all_estimated_ages) / len(all_estimated_ages)
else:
    mean_age = 0.0


if female_count > male_count:
    majority_gender = "Female"
elif male_count > female_count:
    majority_gender = "Male"
else:
    majority_gender = "Equal"

with open(summaryCsvPath, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "total_people",
        "mean_estimated_age",
        "female_count",
        "male_count",
        "majority_gender"
    ])
    writer.writerow([
        total_people,
        f"{mean_age:.2f}",
        female_count,
        male_count,
        majority_gender
    ])



print("[INFO] done.")

print(f"wszystkich ludzi jest: {total_people}")
print(f"sredni wiek to {mean_age:.2f}")
print(f"liczba kobiet {female_count} a liczba mężczyzn {male_count}")
print(f"więcej jest płci: {majority_gender}")