import cv2
import imageio

#face_cascade = cv2.CascadeClassifier(r"C:/Users/Doruk/Anaconda3/envs/tensorflow1/Lib/site-packages/cv2/datahaarcascade-frontalface-default.xml")
#eye_cascade = cv2.CascadeClassifier(r"C:/Users/Doruk/Anaconda3/envs/tensorflow1/Lib/site-packages/cv2/haarcascade-eye.xml")

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = body_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        gray_face = gray[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]
       
    return frame

reader = imageio.get_reader('videos/personel.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('haar_output.avi', fps=fps)
for i, frame in enumerate(reader):
    frame = detect(frame)
    writer.append_data(frame)
    print(i)
writer.close()