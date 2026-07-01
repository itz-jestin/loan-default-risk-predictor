
import streamlit as st
import pickle
import numpy as np

# from PIL import Image

# Load the model
with open("C:\\Users\\asus\\Downloads\\my_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

st.title("Salary Predictor")
st.write("Hello ,let's check your Salary ")
edu_level = st.text_input("Enter Your Education Level")
job = st.text_input("Enter your Job")
years=st.number_input("Enter your Years of Experience") 
# text input
# number input

# uploaded_image = st.file_uploader("Choose an image ... ", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:

# image = Image.open(uploaded_image)
# st.image(image, caption='Uploaded Image', use_column_width=True)



if st.button('Predict'):

# Load the encoder
    with open("C:\\Users\\asus\\Downloads\\le_education.pkl", 'rb') as file:
        le_education = pickle.load(file)


# Load the encoder
    with open("C:\\Users\\asus\\Downloads\\le_job.pkl", 'rb') as file:
        le_job = pickle.load(file)
    user_educ=le_education.transform([edu_level])
    user_job=le_job.transform([job])

    data=[user_educ, user_job, np.array([years])]
    data=np.array(data).reshape(1,-1)
    st.write(loaded_model.predict(data))

#