import os 
import cv2
from PIL import Image
import numpy as np

BASE_DIR = os.getcwd()
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR,'images')
folder_ids = {}
y_labels = []
x_images = []
count = 0

face_cascade = cv2.CascadeClassifier('haarcascades\haarcascades_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root,file)
            folder = os.path.basename(root)
            
            # print(folder,path)

            if not folder in folder_ids:
                folder_ids[folder] = count 
                count += 1
            id = folder_ids[folder]
            size = (550 ,550)
            pil_image = Image.open(path).convert('L') #gray scale
            final_image = pil_image.resize(size)
            image_array = np.array(final_image, 'uint8')
            face = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=1)

            for x,y,w,h in face:
                region = image_array[y:y+h,x:x+w]
                x_images.append(region)
                y_labels.append(id)

# print(x_images,y_labels)


recognizer.train(x_images, np.array(y_labels))
recognizer.save('trainer.yml')