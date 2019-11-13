#!/usr/bin/env python
# coding: utf-8

# ### 0. DB연결

# In[106]:


import pandas as pd
import pymysql
from sqlalchemy import create_engine
import sqlalchemy


# In[107]:


#engine = create_engine("mysql+pymysql://root:"+"mariadb"+"@127.0.0.1:3307/ai_test?charset=utf8", encoding='utf-8')
engine = create_engine("mysql+pymysql://root:"+"mariadb"+"@10.184.9.128:3307/ai_test?charset=utf8", encoding='utf-8')

# Open database connection
db = pymysql.connect(host='10.184.9.128', port=3307, user='root', passwd='mariadb', db='ai_test',charset='utf8',autocommit=True)

conn = engine.connect()

# Connection 으로부터 Cursor 생성
curs = db.cursor()
 
# Connection 닫기
# conn.close()


# In[ ]:





# ### 1. 라이브러리 추가

# In[3]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[52]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


# ### 2. 데이터 정제

# In[5]:


# featureData = pd.read_csv("dataset/OT_F_DATA_V4.csv")


# In[ ]:


# 2분 소요
print("Data Loading.........")
featureData = pd.read_sql_query('SELECT * from TB_ORG_DATA', conn) 


# In[7]:


paramData = pd.read_sql_query('SELECT * from TB_PY_CODE',conn ) 


# In[12]:


le_prod_id = LabelEncoder()


# In[13]:


featureData["LE_PROD_NM"] = le_prod_id.fit_transform(featureData.PROD_NM)


# ### 입력 파라메터 설정

# In[73]:


features = []

for i in paramData[paramData.CODE_ID == 'FEATURE01']['CODE_VALUE'].values:
    features.append(i)
    
label = ['LE_PROD_NM']

features_norm = []

for i in features:
    features_norm.append(i + '_NORM')

# DNN 설정
dnnDensLevel = int(paramData[paramData.CODE_NAME == 'dnnDensLevel']['CODE_VALUE'].values[0])
DensUnit = int(paramData[paramData.CODE_NAME == 'DensUnit']['CODE_VALUE'].values[0])
epochNo = int(paramData[paramData.CODE_NAME == 'epochNo']['CODE_VALUE'].values[0])
DenseActivation = str(paramData[paramData.CODE_NAME == 'DenseActivation']['CODE_VALUE'].values[0])
dnnLoss =  str(paramData[paramData.CODE_NAME == 'dnnLoss']['CODE_VALUE'].values[0])
dnnOptimizer = str(paramData[paramData.CODE_NAME == 'dnnOptimizer']['CODE_VALUE'].values[0])



callback_list = [
    EarlyStopping(monitor="val_loss", patience=3)
]


# In[17]:



##max 값 저장
de_norm = featureData[features].max()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


trainingData_features,testData_features,trainingData_label,testData_label,trainingData_all,testData_all = train_test_split(featureData[features],featureData[label],featureData, test_size=0.2, random_state=1)


# In[20]:


from tensorflow.keras.utils import to_categorical


# In[21]:


trainingData_labels_one = to_categorical(trainingData_label)
testData_labels_one = to_categorical(testData_label)


# In[22]:


def norm(x):
    return (x/x.max())


# In[23]:


trainingData_features_normed = norm(trainingData_features)
testData_features_normed = norm(testData_features)


# ### 일반 화 컬럼 생성 및 전체 데이터 conact

# In[24]:


norm_data  = norm(featureData[features])


# In[25]:


norm_data.columns = features_norm


# In[26]:


featureData_norm = pd.concat([featureData,norm_data],axis=1)


# In[27]:


featureData_norm.head(10)


# ### 3. 모델 생성

# In[28]:


from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import neighbors
from sklearn import decomposition


# In[29]:


inputDim = trainingData_features.loc[0,:].shape
inputDim


# In[30]:


outputShape = len(featureData['LE_PROD_NM'].unique())
outputShape


# In[31]:


model = Sequential()
model.add(Dense(units=DensUnit, activation=DenseActivation,input_shape=inputDim))
for i in range(0,dnnDensLevel):    
    model.add(Dense(units=DensUnit, activation=DenseActivation))
    
model.add(Dense(units=outputShape, activation="softmax"))    
model.summary()


# ### 4. 모델 컴파일

# In[32]:


model.compile(loss=dnnLoss, optimizer=dnnOptimizer, metrics=["accuracy"])


# ### 5. 모델 학습

# In[74]:


model.fit(x=trainingData_features_normed,y=trainingData_labels_one, epochs=epochNo,
          batch_size = 32,
          validation_data=(testData_features_normed,testData_labels_one),
          callbacks = callback_list)


# ### 추론

# In[ ]:


print("predicting .....")


# In[37]:


# predict= pd.DataFrame(model_rf.predict(featureData_norm[features_norm]),columns=["predict_rf"])
featureData['predict_dnn'] = pd.DataFrame(model.predict(featureData_norm[features_norm])).idxmax(axis=1)


# In[38]:


# featureData.to_sql(name='TB_ORG_DATA_PREDICT', con=engine, if_exists='append', index=False)


# In[39]:


featureData.head(10)


# In[40]:


predict= pd.DataFrame(pd.DataFrame(model.predict(testData_features_normed)).idxmax(axis=1),columns=["predict_dnn"])


# In[41]:


test_labels_result = pd.DataFrame(pd.DataFrame(testData_labels_one).idxmax(axis=1),columns=["true_data"])


# In[42]:


accur_score   = accuracy_score(predict['predict_dnn'],test_labels_result['true_data'])


# In[ ]:





# ### 모델 저장

# In[ ]:


print('model saving....')


# In[43]:


currentdate = datetime.now().strftime("%Y%m%d%H%M%S")


# In[44]:


file_path = "d:/log_model/"
model_path = file_path + "model_recommend_dnn_{}.json".format(currentdate)
weight_path = file_path + "model_recommend_dnn_weight_{}.h5".format(currentdate)


# In[45]:


model_json = model.to_json()
with open(model_path, "w") as json_file:
    json_file.write(model_json)

model.save_weights(weight_path)


# ### 수행 이력 저장

# In[ ]:


print('result saving....')


# In[109]:


HisData = pd.read_sql_query('SELECT * from TB_RUN_HISTORY', conn) 
HisData


# In[46]:


# sql = "INSERT INTO TB_RUN_HISTORY (RUN_DATE, RUN_CD, FILE_CD,FILE_PATH,FILE_NAME,REMARK ) VALUES (%s, %s, %s, %s, %s, %s)"
# val = (currentdate,
#        'R' + currentdate  ,
#        '1' ,
#        file_path,
#        model_path,   #+','+ weight_path,
#        str(accur_score)  + ',' + str(features) 
#       )
# curs.execute(sql, val)

# db.commit()


# In[110]:


insertPd = pd.DataFrame([[currentdate,
               'R' + currentdate  ,
               '1' ,
               file_path,
               model_path +','+ weight_path,
               str(paramData[['CODE_NAME','CODE_VALUE']].values),
               str(accur_score)  
                          
              ]],columns=HisData.columns)


# In[111]:


insertPd


# In[112]:


insertPd.to_sql('TB_RUN_HISTORY', conn,if_exists='append',index=False)


# In[113]:


HisData = pd.read_sql_query('SELECT * from TB_RUN_HISTORY', conn)
print(HisData.tail(1))


# In[117]:


print(HisData['FILE_NAME'].values)


# In[ ]:




