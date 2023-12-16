import streamlit as st
from fastai.vision.all import *

# Load the exported model
learn = load_learner('/Users/yassinkissami/Downloads/RosesDataset 2/exported_model.pkl')

# Setting page layout
st.set_page_config(
    page_title="Rose Identification App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))

# Creating main page heading
st.title("Rose Identification App")
st.caption('Upload a photo and click the "Identify Rose" button.')

# Adding image to the first column if image is uploaded
if source_img:
    uploaded_image = PILImage.create(source_img)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

if st.sidebar.button('Identify Rose'):
    # Make predictions using the loaded model
    pred, pred_idx, probs = learn.predict(uploaded_image)
    
    # Display the prediction result
    with st.expander("Identification Result"):
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")
