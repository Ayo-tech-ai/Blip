import os
os.environ["WATCHDOG_OBSERVER"] = "polling"  # prevent Streamlit inotify error
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🌿 ViT-GPT2 Image Captioner",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 ViT-GPT2 Image Describer")
st.write("Upload an image, and this app will generate a detailed caption describing what it sees.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_model()

# -----------------------------
# SETUP DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Caption Generation Button
    if st.button("✨ Generate Description"):
        with st.spinner("Analyzing image... please wait."):
            # Preprocess image
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

            # Generate caption
            output_ids = model.generate(pixel_values, max_length=64, num_beams=4)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Display result
        st.success("✅ Description Generated:")
        st.markdown(f"**{caption}**")

        # Optional: Expand with explanation
        st.info(
            "💡 This description was generated using a vision-language transformer. "
            "It combines a ViT (Vision Transformer) encoder with a GPT-2 decoder trained to describe images in natural language."
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Powered by 🤖 ViT-GPT2 | Built with Streamlit")
