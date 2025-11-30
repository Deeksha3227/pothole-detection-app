import streamlit as st
from PIL import Image
import requests
import io
import base64
import pandas as pd

#  Update this with your ngrok tunnel from Colab
import os
COLAB_NGROK_URL = os.getenv("COLAB_NGROK_URL")


# --- 1. UI Configuration ---
st.set_page_config(page_title="Pothole Detector", layout="centered")
st.title("🚧 Pothole Detection - YOLOv12 & ResNet50")
st.markdown("Upload an image and select a model to run detection from the Colab backend.")

# --- 2. Model Selection ---
MODEL_OPTIONS = ['yolo n v12', 'resnet 50', 'yolo s v12', 'yolo m v12', 'yolo l v12', 'yolo x v12']
selected_model = st.selectbox("**Select Model for Detection:**", MODEL_OPTIONS)

# --- 3. File Upload ---
uploaded_file = st.file_uploader("**Upload a Pothole Image**", type=['png', 'jpg', 'jpeg'])

# --- 4. Function to Connect to Colab Backend ---
def run_pothole_detection_logic(model_name, image_to_process):
    """
    Sends the image to Colab API with selected model name and retrieves results.
    """
    # Ensure image is RGB
    if image_to_process.mode != "RGB":
        image_to_process = image_to_process.convert("RGB")

    # Convert image to byte stream
    img_byte_arr = io.BytesIO()
    image_to_process.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    data = {'model': model_name}

    try:
        response = requests.post(f"{COLAB_NGROK_URL}/detect", files=files, data=data)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Colab API: {e}")

    if response.status_code == 200:
        results = response.json()
        img_data = base64.b64decode(results['detected_image_b64'])
        output_image = Image.open(io.BytesIO(img_data))
        if output_image.mode != "RGB":
            output_image = output_image.convert("RGB")

        return output_image, results.get('accuracy', 'N/A'), results.get('pothole_count', 0)
    else:
        try:
            error_message = response.json().get('error', f"Unknown error ({response.status_code})")
        except:
            error_message = f"Unexpected API response ({response.status_code})"
        raise ConnectionError(error_message)

# --- 5. Main Execution ---
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
    st.image(image_to_process, caption='Uploaded Image', use_column_width=True)

    col_run, col_compare = st.columns(2)
    with col_run:
        run_button = st.button(' RUN DETECTION', use_container_width=True)
    with col_compare:
        compare_button = st.button(' COMPARE ALL MODELS', use_container_width=True)

    # --- Single Model Detection ---
    if run_button:
        with st.spinner(f'Running {selected_model.upper()}... Please wait.'):
            try:
                output_image, accuracy, count = run_pothole_detection_logic(selected_model, image_to_process)
                st.success(' Detection Complete!')
                st.subheader("Detection Results")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Model Used", selected_model.upper())
                    st.metric("Potholes Detected", str(count))
                    st.metric("Model Accuracy", f"{accuracy}%")
                with col2:
                    st.image(output_image, caption=f"Output from {selected_model.upper()}", use_column_width=True)

            except Exception as e:
                st.error(f"Error during detection: {e}")

    # --- Compare All Models ---
    if compare_button:
        st.info(" Running comparison for all models... Please wait.")
        results_data = []

        for model in MODEL_OPTIONS:
            try:
                output_image, accuracy, count = run_pothole_detection_logic(model, image_to_process)
                results_data.append({
                    "Model Name": model.upper(),
                    "Accuracy (%)": accuracy,
                    "Detected Potholes": count
                })
            except Exception as e:
                st.warning(f" {model.upper()} failed: {e}")
                results_data.append({
                    "Model Name": model.upper(),
                    "Accuracy (%)": 0,
                    "Detected Potholes": "Error"
                })

        # Sort by Accuracy descending
        df_results = pd.DataFrame(results_data).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

        st.success(" Comparison Complete!")
        st.subheader(" Model Comparison Results (Descending by Accuracy)")
        st.dataframe(df_results, use_container_width=True)

else:
    st.info(" Upload an image to start detection.")
