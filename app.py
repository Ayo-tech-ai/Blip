import streamlit as st
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Page configuration
st.set_page_config(
    page_title="🧠 Image Captioning with BLIP2",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 BLIP2 Image Captioning App")
st.write("Upload an image, and this AI model will describe what it sees in detail!")

@st.cache_resource
def load_model():
    """Load BLIP2 model and processor once (cached)."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

# Load model
processor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Caption generation
    with st.spinner("Generating detailed description... 🧠"):
        text_prompt = (
            "Describe this image in detail. Focus on visible objects, colors, textures, "
            "and if it's a plant, note any signs of disease, color variation, or unusual features."
        )
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        output = model.generate(
            **inputs,
            max_new_tokens=150,       # More tokens → longer descriptions
            num_beams=5,              # Better quality generation
            length_penalty=2.0,       # Encourage longer sentences
            temperature=1.0           # Balanced creativity
        )
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ Done!")
    st.markdown(f"### 📝 Detailed Description:\n> {caption}")

# Footer
st.markdown("---")
st.caption("Powered by [BLIP2: Bootstrapped Language-Image Pretraining (Salesforce)](https://huggingface.co/Salesforce/blip2-opt-2.7b)")
