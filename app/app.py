import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import torch
import os
import time
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms

# -----------------------------
# Utility to convert image to base64 (for HTML styling)
# -----------------------------
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -----------------------------
# Load your model
# -----------------------------
@st.cache_resource
def load_model():
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load("app/efficientnet_fashion_model.pth", map_location="cpu"))
    model.eval()
    return model

# -----------------------------
# Load class labels
# -----------------------------
@st.cache_data
def load_labels():
    return ["sporty", "vintage", "formal", "casual", "streetwear"]

# -----------------------------
# Image pre-processing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fashion Style Classifier", layout="centered")
st.title("👗 Fashion Style Classifier")
st.write("Upload a fashion image to classify it into one of the predefined styles.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Styled image preview
    img_base64 = image_to_base64(image)
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_base64}"
                 style="border: 4px solid #4CAF50;
                        border-radius: 15px;
                        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
                        width: 80%;
                        max-width: 400px;" />
            <p style="font-size:16px; color: #555;">Uploaded Image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("🧠 Classifying...")

    # Load model and labels with timing
    start_time = time.time()
    model = load_model()
    labels = load_labels()
    st.write(f"✅ Model loaded in `{time.time() - start_time:.2f}` seconds.")

    # Prediction
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 1)

    predicted_label = labels[top_idx.item()]
    confidence = top_prob.item()

    st.success(f"🎉 **Prediction:** `{predicted_label}` with `{confidence:.2%}` confidence.")
