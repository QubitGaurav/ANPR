import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from pipeline import VehicleIntelligencePipeline

# Ensure models load only once
@st.cache_resource
def load_pipeline():
    return VehicleIntelligencePipeline()

def main():
    st.set_page_config(page_title="KnightSight ANPR Dashboard", layout="wide")
    st.title("🚗 Edge-Optimized Vehicle Intelligence")
    st.markdown("Upload an image to test the end-to-end pipeline: Vehicle Detection -> Plate Localization -> ANPR.")

    st.sidebar.header("Settings")
    st.sidebar.markdown("This dashboard showcases the lightweight inference pipeline optimized for edge deployments.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert uploaded file to OpenCV format
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            st.markdown("### Processing...")
            
            with st.spinner("Running Inference Pipeline..."):
                pipeline = load_pipeline()
                results, vehicles, plates = pipeline.process_image(image_array=img_bgr)
                annotated_img = pipeline.annotate_image(img_bgr, results, vehicles)
                
            # Convert back to RGB for Streamlit
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(annotated_rgb, caption="Annotated Result", use_column_width=True)

            st.markdown("### Structured Output")
            st.json(json.dumps(results, indent=4))

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
