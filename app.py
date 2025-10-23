import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Page configuration
st.set_page_config(
    page_title="🧠 Image Captioning with BLIP",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 BLIP Image Captioning App")
st.write("Upload an image, and this AI model will describe what it sees!")

@st.cache_resource
def load_model():
    """Load BLIP model and processor once (cached)."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load model
processor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Caption generation
    with st.spinner("Generating description... 🧠"):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=450)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ Done!")
    st.markdown(f"### 📝 Description:\n> {caption}")

# Footer
st.markdown("---")
st.caption("Powered by [BLIP: Bootstrapped Language Image Pretraining (Salesforce)](https://huggingface.co/Salesforce/blip-image-captioning-base)")
