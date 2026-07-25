
# 🧠 Digit Vision AI Platform (SimpleRNN)

An interactive deep learning web application built with **Streamlit**, **TensorFlow/Keras**, and **OpenCV** that recognizes single handwritten digits (0–9) using a **SimpleRNN (Recurrent Neural Network)** model trained on the standard MNIST dataset.

---

## 🌟 Key Features

- **Recurrent Architecture:** Implements a `SimpleRNN` neural network layer to process pixel sequences line-by-line as sequential time steps.
- **Robust Computer Vision Pipeline:** Features an OpenCV processing pipeline with adaptive binarization, Gaussian blurring, and aspect-ratio preserving bounding box padding.
- **Modern UI/UX:** Styled with custom CSS using dark glassmorphism themes and live diagnostic metrics.
- **Real-Time Diagnostics:** Displays normalized $28 \times 28$ input tensor matrices, prediction certainty ratings, and model architecture details.

---

## ⚙️ Tech Stack

- **Frontend / Interface:** Streamlit
- **Machine Learning Framework:** TensorFlow & Keras
- **Computer Vision Pipeline:** OpenCV (`cv2`), Pillow (PIL)
- **Data Operations:** NumPy

---

## 🏗️ Neural Network Architecture

| Layer | Type | Configuration / Details |
| :--- | :--- | :--- |
| 1 | **SimpleRNN** | 128 Hidden Units, `tanh` Activation |
| 2 | **Dropout** | Rate: 0.1 (Prevents Overfitting) |
| 3 | **Dense** | 64 Units, `relu` Activation |
| 4 | **Dense (Output)** | 10 Classes, `softmax` Activation |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/sathwika9121/digit-recognition-rnn.git](https://github.com/sathwika9121/digit-recognition-rnn.git)
cd digit-recognition-rnn

```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt

```

### 4. Run the Application

```bash
python -m streamlit run app.py

```

---

## 📋 Best Practices for High Accuracy

* **High Contrast:** Write using dark ink on a clean, light or white background.
* **Single Digit:** Ensure only one digit is visible per uploaded image.
* **Centering:** Keep the handwritten character roughly centered within the image frame.

```

---

### 📤 Commands to update your `README.md` on GitHub:

Run these quick commands in your terminal to push the new README to your repository:

```powershell
git add README.md
git commit -m "Docs: Update comprehensive README.md"
git push origin main

```
