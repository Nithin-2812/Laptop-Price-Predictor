import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Background image banner */
    .stApp {
        background-image: url('https://your-image-url-here.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        position: relative;
    }

    /* Overlay to improve text readability */
    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5); /* Black overlay with 50% opacity */
        z-index: -1; /* Place behind the content */
    }

    /* Style for the title */
    .title {
        text-align: center;
        font-size: 3em;
        color: #FFFFFF;
        text-shadow: 2px 2px #000000;
    }

    /* Style for the selectboxes and inputs */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 1.2em;
        color: #FFFFFF;
    }

    /* Style for the button */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.5em;
        padding: 10px;
        border-radius: 8px;
    }

    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add an overlay div
st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title with custom class
st.markdown('<div class="title">Laptop Price Predictor</div>', unsafe_allow_html=True)

# UI elements
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prediction button with custom styling
if st.button('Predict Price'):
    # Feature engineering
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    price = int(np.exp(pipe.predict(query)[0]))
    st.title(f"The predicted price of this configuration is ₹{price}")
