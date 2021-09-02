# -*- coding: utf-8 -*-
"""
Created on Sat Sep 2  2021
@author: RAIL Group
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 2  2021
@author: RAIL Group at Raparin University
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
import numpy as np
from scipy.io import wavfile

import pyaudio
import wave
from PIL import Image

import tensorflow as tf
import sounddevice as sd
import os
from scipy.io.wavfile import write


def record():
    fs = 16000  # Sample rate
    seconds = 1  # Duration of recording
    st.success('start recordeing...')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)
    #st.success('finished recording')

   # global ph
      
    rate, data = wavfile.read('output.wav')
    b=np.array(data[:,0], dtype=np.float64)
    

    b=np.reshape(b, (1, 16000,1)) 
    prediction = classifier.predict(b)
    y_pred = np.argmax(prediction, axis=1)
   # ph=ph*10+y_pred
    st.audio('output.wav', format='wav')
    return y_pred
#app=Flask(__name__)
#Swagger(app)

classifier=tf.keras.models.load_model("predict.model")


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
       



def main():
    st.title("Kurdish Digit Recognition")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Kurdish Digit Recognition </h2>
    </div>
    """
    
    result=""
    if st.button("Predict"):
        
        result=record()
        
        st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("RAIL Group")
        st.text("this is a researcher group ar Raparin University")

if __name__=='__main__':
    main()
    