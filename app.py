import os
os.environ["WATCHDOG_OBSERVER"] = "polling"  # prevent Streamlit inotify error
import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🌿 BLIP Detailed Image Captioner",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 BLIP Detailed Image Describer")
st.write("Upload an image, and this app will generate a comprehensive, detailed description using BLIP model.")

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
# GENERATION SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Generation Settings")

# Length control
max_length = st.sidebar.slider(
    "Maximum description length",
    min_value=50,
    max_value=500,  # Increased maximum
    value=200,
    help="Longer length = more detailed descriptions"
)

# Creativity/quality trade-off
num_beams = st.sidebar.slider(
    "Generation quality (beams)",
    min_value=1,
    max_value=10,
    value=4,
    help="Higher = better quality but slower"
)

# -----------------------------
# PROMPT TEMPLATES
# -----------------------------
prompt_templates = {
    "Default": "a detailed photograph of",
    "Plant Disease Analysis": "a close up of a plant leaf showing detailed symptoms including spots, discoloration, and texture. The leaf appears to have",
    "Technical Description": "a technical detailed description of this image showing",
    "Comprehensive Analysis": "a comprehensive analysis of this image describing colors, textures, composition, and details. The image shows",
    "Scientific Observation": "a scientific observation of this plant leaf detailing visible characteristics. The leaf exhibits"
}

# -----------------------------
# IMAGE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# Prompt selection
template_choice = st.selectbox(
    "🎯 Description Style",
    options=list(prompt_templates.keys()),
    help="Choose a style to guide the description"
)

custom_prompt = st.text_area(
    "✏️ Custom Prompt (Optional)",
    placeholder="E.g.: 'a detailed photograph of a plant leaf showing'",
    height=60,
    help="Write a prompt that starts the description naturally"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Generate multiple descriptions option
    generate_multiple = st.checkbox("Generate multiple variations", value=False)
    
    if st.button("✨ Generate Detailed Description"):
        with st.spinner("Analyzing image in detail... This may take a few seconds."):
            try:
                # Use custom prompt if provided, otherwise use template
                if custom_prompt.strip():
                    final_prompt = custom_prompt
                else:
                    final_prompt = prompt_templates[template_choice]
                
                descriptions = []
                
                if generate_multiple:
                    # Generate 3 different descriptions with varied parameters
                    for i in range(3):
                        inputs = processor(image, text=final_prompt, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, 
                                max_length=max_length,
                                num_beams=num_beams,
                                do_sample=True if i > 0 else False,  # Sample for variations
                                temperature=0.8 + (i * 0.1),  # Vary temperature
                                early_stopping=True,
                                no_repeat_ngram_size=3,
                                length_penalty=1.5  # Strongly encourage longer text
                            )
                        
                        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
                        descriptions.append(caption)
                    
                    # Display multiple descriptions
                    st.success(f"✅ Generated {len(descriptions)} detailed descriptions:")
                    
                    for i, desc in enumerate(descriptions, 1):
                        with st.expander(f"Description {i}", expanded=i==1):
                            st.write(desc)
                            
                            # Show prompt used
                            st.caption(f"Prompt used: '{final_prompt}'")
                
                else:
                    # Generate single detailed description with optimal parameters
                    inputs = processor(image, text=final_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, 
                            max_length=max_length,
                            num_beams=num_beams,
                            do_sample=True,
                            temperature=0.9,
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                            length_penalty=1.8,  # Strong length encouragement
                            repetition_penalty=1.1  # Reduce repetition
                        )
                    
                    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
                    
                    # Display single description
                    st.success("✅ Detailed Description Generated:")
                    st.markdown(f"**{caption}**")
                    
                    # Show stats and prompt used
                    word_count = len(caption.split())
                    st.caption(f"📊 Description length: {len(caption)} characters, {word_count} words")
                    st.caption(f"🎯 Prompt used: '{final_prompt}'")

            except Exception as e:
                st.error(f"❌ Error generating description: {str(e)}")
                st.info("💡 Try using a simpler prompt or reducing the maximum length.")

        # Enhanced model information
        with st.expander("🔧 Generation Parameters Used"):
            st.write(f"""
            - **Max Length**: {max_length}
            - **Number of Beams**: {num_beams}
            - **Prompt Used**: '{final_prompt}'
            - **Device**: {device.upper()}
            """)

# -----------------------------
# SIDEBAR WITH TIPS
# -----------------------------
with st.sidebar:
    st.header("💡 Tips for Better Descriptions")
    st.markdown("""
    **For plant disease analysis:**
    
    🔹 Use plant-specific prompts
    🔹 Set max length to 200-300
    🔹 Use 4-6 beams for balance
    🔹 Try these prompts:
    
    *"a close up of a plant leaf showing detailed symptoms including spots, discoloration, and texture. The leaf appears to have"*
    
    *"a diseased plant leaf exhibiting visible symptoms such as"*
    
    *"scientific analysis of plant health showing"*
    """)
    
    st.header("📊 Current Settings")
    st.write(f"""
    - Max Length: **{max_length}**
    - Beams: **{num_beams}**
    - Device: **{device.upper()}**
    """)

# -----------------------------
# EXAMPLE SECTION
# -----------------------------
with st.expander("📝 Example Prompts for Plant Analysis"):
    st.markdown("""
    **Copy and paste these into the custom prompt field:**
    
    ```
    a close up of a plant leaf showing detailed symptoms including spots, discoloration, texture abnormalities, and overall health condition. The leaf appears to have
    ```
    
    ```
    a scientific analysis of this plant leaf detailing visible disease symptoms, color variations, spot patterns, and physical damage. The leaf exhibits
    ```
    
    ```
    a comprehensive plant health assessment showing detailed observations of leaf condition, including any signs of fungal, bacterial, or viral infections. Visible symptoms include
    ```
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Powered by 🤖 BLIP Image Captioning Large | Built with Streamlit | Enhanced for detailed descriptions")
