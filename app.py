import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==========================================
# SYSTEM & MODEL SETUP
# ==========================================

MODEL_PATH = "rnn_digit_classifier.keras"

def build_and_train_rnn():
    """Trains a SimpleRNN model on MNIST dataset and saves artifact."""
    st.toast("Initial setup: Training model on MNIST...", icon="⚙️")
    
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Scale pixel values to range [0, 1]
    train_x = train_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0

    # Categorical One-Hot Encoding
    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    # Architecture definition
    model = Sequential([
        SimpleRNN(128, input_shape=(28, 28), activation="tanh", return_sequences=False),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Execution phase
    model.fit(
        train_x, 
        train_y, 
        epochs=5, 
        batch_size=64, 
        validation_split=0.15,
        verbose=1
    )

    model.save(MODEL_PATH)
    return model

@st.cache_resource
def get_compiled_model():
    if not os.path.exists(MODEL_PATH):
        return build_and_train_rnn()
    return load_model(MODEL_PATH)

def process_and_infer(input_img, model_ref):
    """Preprocesses input image and generates digit class predictions."""
    # Convert image to grayscale array
    raw_array = np.array(input_img.convert("L"))
    
    # Adaptive resizing & contrast adjustment
    resized = cv2.resize(raw_array, (280, 280), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    # Binarization
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find bounding box around digit
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None, 0.0, None

    x, y, w, h = cv2.boundingRect(coords)
    roi = binary[y:y+h, x:x+w]

    # Target square frame padding
    max_side = max(w, h) + 30
    padded_canvas = np.zeros((max_side, max_side), dtype=np.uint8)
    
    pad_y = (max_side - h) // 2
    pad_x = (max_side - w) // 2
    padded_canvas[pad_y:pad_y+h, pad_x:pad_x+w] = roi

    # Rescale to standard 28x28 array
    final_input = cv2.resize(padded_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    normalized_tensor = (final_input.astype("float32") / 255.0).reshape(1, 28, 28)

    # Model inference
    probs = model_ref.predict(normalized_tensor, verbose=0)[0]
    best_class = int(np.argmax(probs))
    score = float(probs[best_class])

    return best_class, score, final_input


# ==========================================
# STREAMLIT APPLICATION INTERFACE
# ==========================================

st.set_page_config(
    page_title="RNN Digit Studio",
    page_icon="🧠",
    layout="wide"
)

# Dark Slate & Modern Violet Custom Styling
st.markdown("""
<style>
    /* Dark Canvas Background */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    
    /* Hero Section Card */
    .main-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(8px);
    }
    
    .title-text {
        font-size: 30px;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 4px;
    }
    
    .subtitle-text {
        color: #8b949e;
        font-size: 14px;
    }
    
    /* Result Display Badge */
    .prediction-container {
        background: #161b22;
        border: 1px solid #8b5cf6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.15);
    }
    
    .predicted-digit {
        font-size: 72px;
        font-weight: 800;
        color: #a78bfa;
        line-height: 1;
        margin: 10px 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(139, 92, 246, 0.2);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    
    /* Custom Button Style */
    .stButton > button {
        background: #7c3aed;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #6d28d9;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
<div class="main-card">
    <div class="title-text">🧠 RNN Digit Studio</div>
    <div class="subtitle-text">Sequence-Based Handwritten Digit Recognition Engine</div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Dashboard
with st.sidebar:
    st.markdown("### 📊 System Specs")
    st.markdown("<span class='status-badge'>Model Active</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Architecture:** SimpleRNN")
    st.write("**Hidden Units:** 128")
    st.write("**Dense Layers:** 64 → 10")
    st.write("**Input Shape:** 28 × 28 px")
    st.markdown("---")
    st.caption("Framework: Keras / TensorFlow")

# Interactive Layout
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("#### 📤 Input Image")
    file_buffer = st.file_uploader(
        "Upload a handwritten digit scan", 
        type=["png", "jpg", "jpeg"]
    )
    
    if file_buffer:
        raw_image = Image.open(file_buffer)
        st.image(raw_image, caption="Uploaded Source", use_container_width=True)

with col_right:
    st.markdown("#### ⚡ Classification")
    
    if file_buffer is not None:
        model = get_compiled_model()
        
        if st.button("Run Inference Process", use_container_width=True):
            with st.spinner("Analyzing tensor patterns..."):
                predicted_label, probability, preprocessed_matrix = process_and_infer(raw_image, model)
            
            if predicted_label is None:
                st.error("Unable to recognize digit pattern. Please try another image.")
            else:
                # Prediction Card
                st.markdown(f"""
                <div class="prediction-container">
                    <p style="color: #8b949e; margin: 0; font-size: 13px;">PREDICTED RESULT</p>
                    <div class="predicted-digit">{predicted_label}</div>
                    <p style="color: #c4b5fd; margin: 0; font-size: 14px;">Confidence: <b>{probability * 100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(probability)
                
                # Preprocessing Diagnostic Window
                with st.expander("🛠️ Processing Diagnostics"):
                    d1, d2 = st.columns([1, 2])
                    with d1:
                        st.image(preprocessed_matrix, caption="Processed Matrix", width=120)
                    with d2:
                        st.write(f"**Target Class:** {predicted_label}")
                        st.write(f"**Probability:** {probability:.4f}")
                        st.write(f"**Matrix Shape:** {preprocessed_matrix.shape}")
    else:
        st.info("Upload an image on the left to begin classification.")

# Guidance Footer
st.markdown("---")
with st.expander("📌 Image Upload Tips"):
    st.markdown("""
    - Draw digits with dark ink on light/white paper.
    - Keep digits centered within the image frame.
    - Ensure only a single digit is visible per file.
    """)