import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ThyroidProject.utils.common import read_html_file
from pathlib import Path
import pandas as pd
import ydf
import os

# Load the saved model
model = ydf.load_model('model')

data_mapping = {"Male": "M", "Female": "F", True: "t", False: "f"}
target = ['Negative', 'Hypothyroid', 'Hyperthyroid']


# Streamlit App

def main():
    st.title("Thyroid Disease Prediction App")
    st.image("templates/istockphoto-1316440125-612x612.jpg")

    # Button to trigger displaying or hiding HTML
    display_button_drift = st.button("Drift Report")
    display_button_test = st.button("Test Report")

    # File path of your HTML file
    html_file_path_drift = Path(os.path.join("drift_reports", "report.html"))
    html_file_path_test = Path(os.path.join("drift_reports", "test.html"))

    if display_button_drift:
        # Read HTML content
        html_content = read_html_file(html_file_path_drift)

        # Display HTML content
        components.html(html_content, width=1000, height=800, scrolling=True)

    if display_button_test:
        # Read HTML content
        html_content = read_html_file(html_file_path_test)

        # Display HTML content
        components.html(html_content, width=1000, height=800, scrolling=True)

    # Sidebar with user input
    st.sidebar.header("Input Parameters")

    # Collect user input
    age = st.sidebar.slider("Age", 1, 100, 25)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    on_thyroxine = st.sidebar.checkbox("On Thyroxine")
    query_on_thyroxine = st.sidebar.checkbox("Query On Thyroxine")
    on_antihyroid_meds = st.sidebar.checkbox("on antihyroid meds")
    sick = st.sidebar.checkbox("Sick")
    pregnant = st.sidebar.checkbox("Pregnant")
    thyroid_surgery = st.sidebar.checkbox("Undergone Thyroid Surgery")
    I131_treatment = st.sidebar.checkbox("I131 treatment")
    query_hypothyroid = st.sidebar.checkbox("? Hypothyroid")
    query_hyperthyroid = st.sidebar.checkbox("? Hyperthyroid")
    lithium = st.sidebar.checkbox("Lithium")
    goitre = st.sidebar.checkbox("Goitre")
    tumor = st.sidebar.checkbox("Tumor")
    hypopituitary = st.sidebar.checkbox("Hypopituitary")
    psych = st.sidebar.checkbox("Psych")
    TSH = st.sidebar.number_input(
        label="TSH", value=None, placeholder="0.005<=TSH<=530.00")
    T3 = st.sidebar.number_input(
        label="T3", value=None, placeholder="0.05<=T3<=18.00")
    TT4 = st.sidebar.number_input(
        label="TT4", value=None, placeholder="2.00<=TT4<=600.00")
    T4U = st.sidebar.number_input(
        label="T4U", value=None, placeholder="0.170<=TT4<=2.33")
    FTI = st.sidebar.number_input(
        label="FTI", value=None, placeholder="1.4<=FTI<=881")
    submit = st.sidebar.button("Submit")

    if submit:
        cols = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antihyroid_meds', 'sick', 'pregnant',
                'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
                'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        data = [age, data_mapping.get(sex), data_mapping.get(on_thyroxine), data_mapping.get(query_on_thyroxine),
                data_mapping.get(on_antihyroid_meds), data_mapping.get(
                    sick), data_mapping.get(pregnant),
                data_mapping.get(thyroid_surgery), data_mapping.get(
                    I131_treatment),
                data_mapping.get(query_hypothyroid), data_mapping.get(
                    query_hyperthyroid), data_mapping.get(lithium),
                data_mapping.get(goitre), data_mapping.get(
                    tumor), data_mapping.get(hypopituitary),
                data_mapping.get(psych), TSH, T3, TT4, T4U, FTI]
        df = pd.DataFrame(np.array(data).reshape(1, len(data)), columns=cols)
        preds = model.predict(df)
        result = int(np.argmax(preds, axis=1))

        st.subheader("Prediction:")
        st.write(
            f"The predicted thyroid disease class is: {target[result].upper()}")


if __name__ == "__main__":
    main()
