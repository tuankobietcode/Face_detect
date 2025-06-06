from uu import encode

import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from openpyxl import Workbook, load_workbook
import csv
# Load ảnh
path = "pic2"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(len(images))
    print(classNames)

# Mã hóa ảnh
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknow = Mahoa(images)
print("Mã hóa thành công")
print(len(encodeListknow))

# Danh sách tên đã nhận diện (chỉ ghi 1 lần trong 1 phiên chạy)
recognized_names = set()

# Thêm data nhận được vào Log




def Add_data(name):
    if name in recognized_names:
        return

    now = datetime.now().strftime('%Y-%m-%d ,%H:%M:%S')
    file = "log.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Thời gian", " tên"])

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([now, name])

    recognized_names.add(name)


# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame, (0, 0), None, fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # Xác định vị trí khuôn mặt
    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):
        matches = face_recognition.compare_faces(encodeListknow, encodeFace)
        faceDif = face_recognition.face_distance(encodeListknow, encodeFace)
        matchIndex = np.argmin(faceDif)

        if faceDif[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            Add_data(name)
        else:
            name = "Unknow"

        # In tên lên frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 225), 2)

    cv2.imshow(".", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
