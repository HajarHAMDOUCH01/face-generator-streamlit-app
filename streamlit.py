import torch
from torchvision.utils import save_image
from torchvision import transforms
import sys, os, io
import streamlit as st
from PIL import Image
import numpy as np

sys.path.append("../")
from model.style_gan import StyleGAN

from pathlib import Path

def load_stylegan_model(model_path, device):
    model = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")
    model = model.to(device)
    return model

def generate_faces(num_faces, model, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_faces, 512, device=device)
        images = model.generate(z, truncation_psi=0.65)
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    return images

def tensor_to_pil(tensor):
    """Convert a single image tensor to PIL Image"""
    # tensor shape: [C, H, W]
    img_array = tensor.cpu().numpy()
    img_array = np.transpose(img_array, (1, 2, 0))  # [H, W, C]
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def main():
    st.title(" Face Generator with StyleGAN")
    st.write("Generate realistic faces using StyleGAN2-FFHQ model")
    
    model_path = "model/model.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (cache to avoid reloading on every interaction)
    @st.cache_resource
    def load_cached_model():
        return load_stylegan_model(model_path, device)
    
    with st.spinner("Loading model..."):
        model = load_cached_model()
    
    st.success(f"Model loaded on {device}")
    
    # User input for number of faces
    st.header("Choose number of generated faces")
    num_faces = st.slider("Number of faces to generate:", min_value=1, max_value=10, value=4)
    
    # Generate button
    if st.button(" Generate New Images", type="primary"):
        with st.spinner(f"Generating {num_faces} face(s)..."):
            # Generate images
            images = generate_faces(num_faces, model, device)
            
            # Display images in a grid
            st.subheader(f"Generated Faces ({num_faces})")
            
            # Create columns for grid layout (3 images per row)
            cols_per_row = 3
            for i in range(0, num_faces, cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < num_faces:
                        with cols[j]:
                            # Convert tensor to PIL image
                            pil_image = tensor_to_pil(images[idx])
                            
                            # Display image
                            st.image(pil_image, caption=f"Face {idx + 1}", use_container_width=True)

                            # Download button for each image
                            buf = io.BytesIO()
                            pil_image.save(buf, format="PNG")
                            st.download_button(
                                label=f"⬇️ Download",
                                data=buf.getvalue(),
                                file_name=f"generated_face_{idx + 1}.png",
                                mime="image/png",
                                key=f"download_{idx}"
                            )
    
    with st.expander("About this model"):
        st.write("""
        This app uses StyleGAN2 trained on FFHQ dataset to generate 128x128 face images.
        
        - **Model**: StyleGAN2-FFHQ-128
        - **Resolution**: 128 × 128 pixels
        - **Truncation PSI**: 0.65 (controls image quality vs diversity)
        """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "🤗 View Model on Hugging Face",
            "https://huggingface.co/hajar001/stylegan2-ffhq-128",
            use_container_width=True
        )

if __name__ == "__main__":
    main()