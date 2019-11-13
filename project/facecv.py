#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.keras.models import model_from_json


# In[3]:


EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise"]
EMOTIONS_LIST2 = ["Angry", "Angry", "Fear", "Happy","Neutral", "Sad", "Neutral"]

GENDER_LIST = ["MAN", "WOMAN"]
## wiki crop 데이터는 남여가 숫자가 반대
GENDER_WIKILIST = ["WOMAN", "MAN"]


# In[4]:


json_file = open("./model_emotion.json","r")
loaded_model_json = json_file.read()
json_file.close()


# In[5]:


loaded_model = model_from_json(loaded_model_json)


# In[6]:


loaded_model.load_weights("./model_emotion_kopo_weight.h5")
# testImg = cv2.imread("d:\sample_image.PNG",cv2.IMREAD_COLOR)


# In[7]:


gen_json_file = open("./model_gender.json","r")
gen_loaded_model_json = gen_json_file.read()
gen_json_file.close()


# In[8]:


gen_loaded_model = model_from_json(gen_loaded_model_json)


# In[9]:


gen_loaded_model.load_weights("./model_gender_kopo_weight.h5")


# In[10]:


age_json_file = open("./model_age.json","r")
age_loaded_model_json = age_json_file.read()
age_json_file.close()


# In[11]:


age_loaded_model = model_from_json(age_loaded_model_json)


# In[12]:


age_loaded_model.load_weights("./model_age_kopo_weight.h5")


# In[13]:


import cv2
import sys


font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[14]:


cap = cv2.VideoCapture(0)


while(True):
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        break;
    
    
    key = cv2.waitKey(33)
    ##얼굴 찾기
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5,
                 minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 획득 이미지 회색컬러로 변경
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 인식된 얼굴 주위에 사각형 영역 표시
    # haar cascading 통해 face 이미지 추출 후 좌표활용)
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        # 획득 이미지 크기조절 및 딥러닝 입력 형태로 변환
        roi = cv2.resize(fc, (48, 48))
        roip = roi.reshape(1,48,48,1)
        roip = roip/255.0
             
        # 예측 감정
        predict = loaded_model.predict(roip)
        # 예측 성별
        genpredict = gen_loaded_model.predict(roip)
        # 나이 예측
        agepredict = age_loaded_model.predict(roip)
        
        # 예측결과 리턴
        pred = EMOTIONS_LIST2[np.argmax(predict)]
        ##성별
        genpred = GENDER_LIST[np.argmax(genpredict)]
        #wiki 크롭은 반대로 남여 순서 반대
        #genpred = GENDER_WIKILIST[np.argmax(genpredict)]
        
        #나이
        agepred = np.argmax(agepredict)
        
        # 예측결과 화면 시연 및 사각형 영역 표시
        cv2.putText(frame, pred+' ' + genpred + ' ' + str(agepred), (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    
    cv2.imshow('color_frame', frame)
    
    if key==27:
        # esc key
        break
        cap.release()
        cv2.destroyAllWindows()
    
    elif key==32:
        cv2.imwrite("d:/capture_rec_images.jpg", frame )
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




