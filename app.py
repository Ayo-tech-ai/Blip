import os
os.environ["WATCHDOG_OBSERVER"] = "polling"  # prevent Streamlit inotify error
import streamlit as st
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🌿 BLIP Image Captioner",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 BLIP Image Describer")
st.write("Upload an image, and this app will generate a detailed caption describing what it sees using BLIP model.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    # Load BLIP model and processor
    model_name = "Salesforce/blip-image-captioning-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# -----------------------------
# SETUP DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# Optional: Add custom prompt
custom_prompt = st.text_input(
    "Optional: Add a custom prompt prefix",
    placeholder="e.g., 'a close up of a plant leaf showing'",
    help="Add context to guide the caption generation"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Caption Generation Button
    if st.button("✨ Generate Description"):
        with st.spinner("Analyzing image with BLIP... please wait."):
            try:
                # Generate caption with optional custom prompt
                if custom_prompt:
                    # Conditional image captioning with prompt
                    inputs = processor(image, text=custom_prompt, return_tensors="pt").to(device)
                else:
                    # Unconditional image captioning
                    inputs = processor(image, return_tensors="pt").to(device)
                
                # Generate caption
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, 
                        max_length=100, 
                        num_beams=5,
                        early_stopping=True
                    )
                
                caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

                # Display result
                st.success("✅ Description Generated:")
                st.markdown(f"**{caption}**")

                # Show what prompt was used
                if custom_prompt:
                    st.info(f"📝 Used prompt: '{custom_prompt}'")

            except Exception as e:
                st.error(f"❌ Error generating description: {str(e)}")

        # Optional: Expand with explanation
        with st.expander("ℹ️ About this model"):
            st.markdown("""
            **BLIP (Bootstrapping Language-Image Pre-training)** is a vision-language model that:
            - Excels at both understanding and generating text from images
            - Uses a ViT-Large backbone for visual encoding
            - Was trained on large-scale web image-text pairs
            - Particularly good at detailed, contextual image descriptions
            
            **Model**: `Salesforce/blip-image-captioning-large`  
            **Parameters**: ~500 million  
            **Training**: COCO dataset + web data
            """)

# -----------------------------
# SIDEBAR WITH EXAMPLE PROMPTS
# -----------------------------
with st.sidebar:
    st.header("💡 Prompt Examples")
    st.markdown("""
    Try these prompts for better results:
    
    **General:**
    - `a photography of`
    - `a detailed description of`
    
    **Plant-focused:**
    - `a close up of a plant leaf showing`
    - `symptoms of plant disease including`
    - `this diseased leaf exhibits`
    
    **Object-focused:**
    - `a product photo of`
    - `an architectural photo of`
    """)
    
    st.header("⚙️ Model Info")
    st.markdown(f"""
    - **Device**: {device.upper()}
    - **Model**: BLIP Large
    - **Status**: ✅ Loaded
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Powered by 🤖 BLIP Image Captioning Large | Built with Streamlit")
