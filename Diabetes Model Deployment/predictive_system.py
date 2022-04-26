# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('D:/ShapeAI/AI ML Projects/Diabetes Model Deployment/trained_model.sav', 'rb')) #rb-read binary

  
  
input_data = (1,103,30,38,83,43.3,0.183,33)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have Diabetes')
else:
  print('The Person has Diabetes')