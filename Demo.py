#File này chỉ demo so sánh ảnh sử dụng các hàm của face_recognition
import cv2
import face_recognition
from face_recognition import face_locations
# Load ảnh từ file pic 2 có sẵn trong thư mục 
imgElon=face_recognition.load_image_file(r'pic2/elon-musk.jpeg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgCheck=face_recognition.load_image_file(r'pic2/Eloncheck.jpg')
imgCheck=cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)

#thực hiện xác định vị trí khuôn mặt và mã hóa 

faceLoc=face_recognition.face_locations(imgElon)[0]

encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(225,0,225),2)

faceCheck=face_recognition.face_locations(imgElon)[0]
encodeCheck=face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(225,0,225),2)

#So sánh dùng hàm có sẵn
results=face_recognition.compare_faces([encodeElon],encodeCheck)
print(results)

faceDif=face_recognition.face_distance([encodeElon],encodeCheck)
print(faceDif)
cv2.putText(imgCheck,f"{results}{1-round(faceDif[0],2)}",(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,225),2)

cv2.imshow("Elon-check",imgCheck)
cv2.imshow("Elon",imgElon)
cv2.waitKey()

