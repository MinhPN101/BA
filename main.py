import streamlit as st
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
def get_is_holiday(day, month):
    if month == 12 and 10 < day < 30:
        return 1
    elif month == 11 and 15 < day < 30:
        return 1
    elif month == 9 and 1 < day < 15:
        return 1
    elif month == 2 and 1 < day < 15:
        return 1
    else:
        return 0
def get_type(str_type):
    if str_type == 'A':
        return 1
    elif str_type == 'B':
        return 2
    else:
        return 3
st.title("Walmart Sales Prediction")
st.header("Simple tool for store managers to estimate future sale for their store")

demo_data = pd.read_csv('data/train.csv')
st.sidebar.header("User Input")
store = st.sidebar.selectbox("Store", list(range(1, 46)))
dept = st.sidebar.selectbox("Department", list(range(1, 100)))
type = st.sidebar.selectbox("Type", ['A', 'B', 'C'])
type_int = get_type(type)
with st.sidebar.form("Datetime"):
    year = st.selectbox("Year", list(range(2013, 2023)))
    month = st.selectbox("Month", list(range(1, 13)))
    day = st.selectbox("Day", list(range(1, 32)))
    is_holiday = get_is_holiday(day, month)
    submit = st.form_submit_button("Enter")
temp = st.sidebar.slider("Temperature", 0.0, 100.0)
cpi = st.sidebar.slider("CPI", 100.0, 250.0)
unemp_rate = st.sidebar.slider("Unemployment rate", 0.0, 20.0)
fuel_price = st.sidebar.slider("Fuel Price", 2.5, 4.5)
size = st.sidebar.slider("Size", 30000, 250000)

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
if st.sidebar.button("Predict sale"):
    feats = [store, dept, temp, fuel_price, cpi, unemp_rate, type_int, size, is_holiday]
    pred = loaded_model.predict([feats])
    st.write("Predicted Sale: ", pred[0])
    # st.write('Predicted Sale: ', 200000*random.random())
if st.sidebar.button("View historical data"):
    sales_df = demo_data[(demo_data['Store'] == store) & (demo_data['Dept'] == dept)]
    # sales_df.index = sales_df.Date
    st.line_chart(sales_df['Weekly_Sales'])
    st.write('**Historical sales for department {} of store {}**'.format(dept, store))