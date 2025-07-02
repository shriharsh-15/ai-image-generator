import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import io
import json

st.title("AI Image Generator")

prompt = st.text_input("Enter your image prompt:")
generate = st.button("Generate Image")

if 'pipe' not in st.session_state:
    st.session_state.pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        st.session_state.pipe = st.session_state.pipe.to("cuda")

if generate and prompt:
    with st.spinner("Generating image..."):
        image = st.session_state.pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        # Save image to buffer
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), file_name="generated.png", mime="image/png")
        # Save attributes
        attributes = {
            "prompt": prompt,
            "model": "runwayml/stable-diffusion-v1-5"
        }
        attr_buf = io.StringIO()
        json.dump(attributes, attr_buf)
        st.download_button("Download Attributes", attr_buf.getvalue(), file_name="attributes.json", mime="application/json")
