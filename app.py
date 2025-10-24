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
    max_value=300,
    value=150,
    help="Longer length = more detailed descriptions"
)

# Creativity/quality trade-off
num_beams = st.sidebar.slider(
    "Generation quality (beams)",
    min_value=1,
    max_value=10,
    value=7,
    help="Higher = better quality but slower"
)

temperature = st.sidebar.slider(
    "Creativity (temperature)",
    min_value=0.1,
    max_value=1.5,
    value=0.9,
    help="Higher = more creative, Lower = more focused"
)

# -----------------------------
# PROMPT TEMPLATES
# -----------------------------
prompt_templates = {
    "Default": "",
    "Detailed Analysis": "Provide a detailed analysis of this image describing:",
    "Plant/Focus": "A comprehensive description of this plant leaf including color, texture, spots, and overall health:",
    "Technical Description": "A technical, detailed description of this image covering composition, colors, objects, and context:",
    "Storytelling": "Describe this image in a vivid, narrative style with rich details:",
    "Scientific Observation": "As a scientific observer, provide detailed observations about this image:"
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
    placeholder="E.g.: 'Describe this image in extreme detail, focusing on colors, textures, objects, lighting, and overall mood:'",
    height=80,
    help="Write a detailed prompt to get more comprehensive descriptions"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)  # FIXED: use_container_width instead of use_column_width

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
                    # Generate 3 different descriptions
                    for i in range(3):
                        if final_prompt:
                            inputs = processor(image, text=final_prompt, return_tensors="pt").to(device)
                        else:
                            inputs = processor(image, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, 
                                max_length=max_length,
                                num_beams=num_beams,
                                temperature=temperature,
                                do_sample=True,  # Enable sampling for variety
                                early_stopping=True,
                                no_repeat_ngram_size=2,
                                length_penalty=1.2  # Encourage longer sequences
                                # REMOVED: bad_words_ids parameter causing error
                            )
                        
                        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
                        descriptions.append(caption)
                    
                    # Display multiple descriptions
                    st.success(f"✅ Generated {len(descriptions)} detailed descriptions:")
                    
                    for i, desc in enumerate(descriptions, 1):
                        with st.expander(f"Description {i}", expanded=i==1):
                            st.write(desc)
                
                else:
                    # Generate single detailed description
                    if final_prompt:
                        inputs = processor(image, text=final_prompt, return_tensors="pt").to(device)
                    else:
                        inputs = processor(image, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, 
                            max_length=max_length,
                            num_beams=num_beams,
                            temperature=temperature,
                            do_sample=True,
                            early_stopping=True,
                            no_repeat_ngram_size=2,
                            length_penalty=1.2
                            # REMOVED: bad_words_ids parameter causing error
                        )
                    
                    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
                    
                    # Display single description
                    st.success("✅ Detailed Description Generated:")
                    st.markdown(f"**{caption}**")
                    
                    # Show stats
                    word_count = len(caption.split())
                    st.caption(f"📊 Description length: {len(caption)} characters, {word_count} words")

            except Exception as e:
                st.error(f"❌ Error generating description: {str(e)}")

        # Enhanced model information
        with st.expander("🔧 Generation Parameters Used"):
            st.write(f"""
            - **Max Length**: {max_length}
            - **Number of Beams**: {num_beams}
            - **Temperature**: {temperature}
            - **Prompt Used**: '{final_prompt if final_prompt else 'Unconditional'}'
            - **Device**: {device.upper()}
            """)

# -----------------------------
# SIDEBAR WITH TIPS
# -----------------------------
with st.sidebar:
    st.header("💡 Tips for Better Descriptions")
    st.markdown("""
    **For more detailed descriptions:**
    
    🔹 **Use longer max length** (200-300)
    🔹 **Increase beam search** (7-10)
    🔹 **Use descriptive prompts**
    🔹 **Try different temperature** (0.7-1.2)
    
    **Effective Prompts:**
    - "A highly detailed description of..."
    - "Describe every element you see including..."
    - "Provide a comprehensive analysis of..."
    - "As an expert observer, describe in detail..."
    """)
    
    st.header("📊 Current Settings")
    st.write(f"""
    - Max Length: **{max_length}**
    - Beams: **{num_beams}**
    - Temperature: **{temperature}**
    - Device: **{device.upper()}**
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Powered by 🤖 BLIP Image Captioning Large | Built with Streamlit | Enhanced for detailed descriptions")
