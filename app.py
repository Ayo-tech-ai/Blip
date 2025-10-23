import os
os.environ["WATCHDOG_OBSERVER"] = "polling"  # prevent Streamlit inotify error

import streamlit as st
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

# Streamlit page setup
st.set_page_config(
    page_title="🌿 InstructBLIP Image Describer",
    page_icon="🧠",
    layout="centered"
)

st.title("🌿 InstructBLIP Image Describer")
st.write("Upload an image, and this app will generate a detailed, intelligent description based on what it sees.")

@st.cache_resource
def load_model():
    """Load InstructBLIP model and processor (cached)."""
    model_name = "Salesforce/instructblip-flan-t5-base"
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Select description style
    description_mode = st.radio(
        "🗒️ Choose description style:",
        ["General Description", "Detailed Scientific (Plant Focused)"]
    )

    if description_mode == "General Description":
        prompt = "Describe this image in clear, detailed English sentences."
    else:
        prompt = (
            "Describe this plant leaf in detail, focusing on visible color, texture, shape, "
            "and any signs of disease such as spots, discoloration, wilting, or curling."
        )

    with st.spinner("Generating description... 🧠"):
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=5,
            length_penalty=1.5,
            repetition_penalty=1.2
        )
        caption = processor.decode(outputs[0], skip_special_tokens=True)

    st.success("✅ Description Generated!")
    st.markdown(f"### 📝 {description_mode}\n> {caption}")

st.markdown("---")
st.caption("🚀 Powered by [Salesforce InstructBLIP](https://huggingface.co/Salesforce/instructblip-flan-t5-base)")
