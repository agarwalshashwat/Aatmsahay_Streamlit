"""
Created on Mon May 10 3:40:41 2021

@author: Shashwat Agarwal 
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import streamlit as st

from PIL import Image


#web = Flask(__name__)

rdf_model = pickle.load(open('rdf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
tfvec = pickle.load(open('tfvec.pkl', 'rb'))



#@web.route('/')
def home():
    return render_template('home.html')




#@web.route('/predict',methods=['POST'])
def  predict(Symptom_1,Symptom_2,Symptom_3):
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in (Symptom_1,Symptom_2,Symptom_3)]
    int_features = [' '.join(int_features)]
    rdf = le.inverse_transform(rdf_model.predict(tfvec.transform(int_features).toarray()))
    nb = le.inverse_transform(nb_model.predict(tfvec.transform(int_features).toarray()))
    knn = le.inverse_transform(knn_model.predict(tfvec.transform(int_features).toarray()))
    
    return (rdf[0],nb[0],knn[0])


 #   return render_template('home.html', prediction_text='The disease you might be having according to the three models i.e; by  random forest classifier is {}, by naive bayes classifier is {} and at last by k-nearest neighbour is {}.'.format(rdf[0],nb[0],knn[0]))
     

def main():
    st.title('Aatmsahay: Your Home Doc')
    html_temp = '''
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Enter your symptoms down below</h2>
    </div>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    Symptom_1 = str(st.text_input('Primary Symptom','Type Here'))
    Symptom_2 = str(st.text_input('Secondary Symptom', 'Type Here'))
    Symptom_3 = str(st.text_input('Tertiary Symptom','Type Here'))
    result = ''
    a = ''
    b = ''
    c = ''
    if st.button("Predict"):
        result = predict(Symptom_1,Symptom_2,Symptom_3)
        a = result[0]
        b = result[1]
        c = result[2]
    st.success('The disease you might be having according to the three models i.e; by  random forest classifier is {}, by naive bayes classifier is {} and at last by k-nearest neighbour is {}.'.format(a,b,c))
    if st.button("About"):
        st.text("This is just a prototype, when in need please consult a doctor.")
        st.text("Built by Shashwat Agarwal")
    
if __name__ == "__main__":
    main()
