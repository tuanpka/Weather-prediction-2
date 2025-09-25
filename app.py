# app.py 

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
import pandas as pd

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("WeatherInferenceApp").getOrCreate()

@st.cache_resource
def load_model(model_path="models/Linear_Regression_model"):
    try:
        model = LinearRegressionModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi load mô hình. Check model kaggle{e}")
        return None

spark = get_spark_session()
model = load_model()

st.title("Dự báo nhiệt độ - thời tiết")
st.write("Nhập các thông số để nhận dự báo nhiệt độ.")

st.sidebar.header("Thông số dự báo")

def user_input_features():
    height_sta = st.sidebar.slider('Độ cao trạm (m)', 50, 500, 150)
    dd = st.sidebar.slider('Hướng gió (°)', 0, 360, 180)
    ff = st.sidebar.slider('Tốc độ gió (m/s)', 0.0, 30.0, 5.0)
    precip = st.sidebar.slider('Lượng mưa (kg/m²)', 0.0, 10.0, 0.0)
    hu = st.sidebar.slider('Độ ẩm (%)', 0, 100, 70)
    td = st.sidebar.slider('Điểm sương (K)', 270.0, 300.0, 285.0)
    hours = st.sidebar.slider('Giờ trong ngày (0-23)', 0, 23, 12)
    days = st.sidebar.slider('Ngày trong năm (1-365)', 1, 365, 180)
    years = st.sidebar.slider('Năm', 2024, 2030, 2025)

    data = {
        'height_sta': height_sta, 'dd': dd, 'ff': ff, 'precip': precip,
        'hu': hu, 'td': td, 'hours': hours, 'days': days, 'years': years
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button("Dự báo"):
    if model:
        spark_input_df = spark.createDataFrame(input_df)

        feature_cols = ["height_sta", "dd", "ff", "precip", "hu", "td", "hours", "days", "years"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        transformed_input = assembler.transform(spark_input_df)
        prediction = model.transform(transformed_input)
        
        predicted_temp_k = prediction.select("prediction").collect()[0]['prediction']

        st.success(f"Dự báo thành công!")
        st.metric(label="Nhiệt độ dự báo (Độ K)", value=f"{predicted_temp_k:.2f} K")
        st.metric(label="Nhiệt độ dự báo (Độ C)", value=f"{(predicted_temp_k - 273.15):.2f} °C")