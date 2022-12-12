import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


## Load Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler =  pickle.load(open('scaling.pkl', 'rb'))


def predict_streamlit(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT):
    new_data = scaler.transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    prediction=regmodel.predict(new_data)
    print(prediction)
    return prediction



def main():
    #st.title("House Price Prediction")
    html_temp = """
    <div style="background-color:DarkCyan;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    CRIM = st.text_input("CRIM")
    ZN = st.text_input("ZN")
    INDUS = st.text_input("INDUS")
    CHAS = st.text_input("CHAS")
    NOX = st.text_input("NOX")
    RM = st.text_input("RM")
    AGE = st.text_input("AGE")
    DIS = st.text_input("DIS")
    RAD = st.text_input("RAD")
    TAX = st.text_input("TAX")
    PTRATIO = st.text_input("PTRATIO")
    B = st.text_input("B")
    LSTAT = st.text_input("LSTAT")

    result=""
    if st.button("Predict"):
        result=predict_streamlit(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("This is built to predict the house prices using Boston Dataset.")

if __name__=='__main__':
    main()