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
    <h2 style="color:white;text-align:center;">Select your symptoms down below</h2>
    </div>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    Symptom_1 = st.selectbox('Primary Symptom',{'Select your primary symptom':0,'Abdominal Pain':'abdominal_pain','Abnormal Menstruation':'abnormal_menstruation','Acidity':'acidity','Acute Liver Failure':'acute_liver_failure','Altered Sensorium':'altered_sensorium','Anxiety':'anxiety','Back Pain':'back_pain','Belly Pain':'belly_pain','Blackheads':'blackheads','Bladder Discomfort':'bladder_discomfort','Blister':'blister','Blood in Sputum':'blood_in_sputum','Bloody Stool':'bloody_stool','Blurred and Distorted Vision':'blurred_and_distorted_vision','Breathelessness':'breathlessness','Brittle Nails':'brittle_nails','Bruising':'bruising','Burning Micturition':'burning_micturition','Chest Pain':'chest_pain','Chills':'chills','Cold hands and feets':'cold_hands_and_feets','Coma':'coma','Congestion':'congestion','Constipation':'constipation','Continuous feel of Urine':'continuous_feel_of_urine','Continuous Sneezing':'continuous_sneezing','Cough':'cough','Cramps':'cramps','Dark Urine':'dark_urine','Dehydration':'dehydration','Depression':'depression','Diarrhoea':'diarrhoea','Dischromic Patches':'dischromic_patches','Distention of Abdomen':'distention_of_abdomen','Dizziness':'dizziness','Drying and Tingling Lips':'drying_and_tingling_lips','Enlarged Thyroid':'enlarged_thyroid','Excessive Hunger':'excessive_hunger','Extra Marital Contacts':'extra_marital_contacts','Family History':'family_history','Fast Heart Rate':'fast_heart_rate','Fatigue':'fatigue','Fluid Overload':'fluid_overload','Foul Smell of Urine':'foul_smell_ofurine','Headache':'headache','High Fever':'high_fever','Hip Joint Pain':'hip_joint_pain','History of Alcohol Consumption':'history_of_alcohol_consumption','Increased Appetite':'increased_appetite','Indigestion':'indigestion','Inflammatory Nails':'inflammatory_nails','Internal Itching':'internal_itching','Irregular Sugar Level':'irregular_sugar_level'})
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
        st.text("This is just a prototype, don't go by the predictions made from this app, when in need please consult a doctor.")
        st.text("Built by Shashwat Agarwal")
    
if __name__ == "__main__":
    main()
