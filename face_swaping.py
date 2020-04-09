
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.namedWindow('image title', cv2.WINDOW_NORMAL)

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    print(faces[0], faces[1])
    
    cropped_face1 = cv2.resize(img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]], (faces[1][3], faces[1][2]))
    
    cropped_face2 = cv2.resize(img[faces[1][1]:faces[1][1]+faces[1][3], faces[1][0]:faces[1][0]+faces[1][2]], (faces[0][3], faces[0][2]))
    
    img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] = cropped_face2
    img[faces[1][1]:faces[1][1]+faces[1][3], faces[1][0]:faces[1][0]+faces[1][2]] = cropped_face1
    
    # Crop all faces found
#     for (x,y,w,h) in faces:
#         cropped_face = img[y:y+h, x:x+w]
#         print(cropped_face.shape)

    return img

frame = cv2.imread('two_face.jpg')
faces = face_extractor(frame)
if faces is not None:
    cv2.imshow('image title', faces)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

# frame = cv2.imread('two_face1.jpg')
# if face_extractor(frame) is not None:
# #     face = cv2.resize(face_extractor(frame), (200, 200))
# #     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     faces = cv2.resize(face_extractor(frame), (200, 200))
#     for face in faces:
#         print(1)


#     # Put count on images and display live count
# #     cv2.imshow('Face Cropper', face)
# #     cv2.waitKey()
# #     cv2.destroyAllWindows()
# else:
#     print("Face not found")
#     pass


# cv2.destroyAllWindows()      
# print("Collecting Samples Complete")


# In[ ]:


# import cv2
# frame = cv2.imread('two_face1.jpg')
# frame = frame[982:1520, 342:880]
# cv2.imshow('image title', frame)
# cv2.waitKey()
# cv2.destroyAllWindows()

