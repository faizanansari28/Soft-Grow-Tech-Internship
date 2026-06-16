# 🔢 Project 2 – Handwritten Digit Recognition

> **SoftGrowTech Internship | Task 2**

A Convolutional Neural Network (CNN) trained on the **MNIST dataset** to recognize handwritten digits (0–9) with high accuracy, built using TensorFlow/Keras.

---

## ✨ Features

- CNN architecture: Conv → Pool → Conv → Pool → Flatten → Dense
- Trained on **70,000 MNIST images** (60k train / 10k test)
- **Early stopping** to prevent overfitting
- Achieves **~99% test accuracy**
- Predicts random test samples with confidence scores
- Saves trained model to disk (`.keras` format) for reuse
- Bonus snippet to load saved model and predict a single image

---

## 🛠️ Technologies Used

| Library              | Purpose                          |
|----------------------|----------------------------------|
| `tensorflow` / `keras` | Model building & training      |
| `numpy`              | Array manipulation               |
| MNIST Dataset        | Built-in via `keras.datasets`    |

---

## ⚙️ Installation

```bash
pip install tensorflow numpy
```

MNIST dataset is downloaded automatically by Keras on first run.

---

## ▶️ How to Run

```bash
python project2_digit_recognition.py
```

---

## 🧠 Model Architecture

```
Input (28×28×1)
    │
    ├── Conv2D(32 filters, 3×3, ReLU)    ← Block 1
    ├── MaxPooling2D(2×2)
    │
    ├── Conv2D(64 filters, 3×3, ReLU)    ← Block 2
    ├── MaxPooling2D(2×2)
    │
    ├── Flatten
    ├── Dropout(0.4)                      ← Regularisation
    ├── Dense(128, ReLU)
    └── Dense(10, Softmax)                ← Output: digits 0–9
```

| Parameter   | Value                          |
|-------------|--------------------------------|
| Optimizer   | Adam                           |
| Loss        | Sparse Categorical Crossentropy |
| Batch size  | 128                            |
| Max epochs  | 10 (with early stopping)       |
| Validation  | 10% of training data           |

---

## 📋 How It Works

```
[1/4] Load & normalize MNIST data  (pixel values 0–255 → 0.0–1.0)
[2/4] Build CNN model
[3/4] Train with early stopping on val_accuracy
[4/4] Evaluate on test set + predict 10 random samples
```

---

## 🖥️ Sample Output

```
  Index   True Label    Predicted   Confidence   Correct?
  -------------------------------------------------------
   4821            7            7       99.87%          ✓
   1042            3            3       99.54%          ✓
    305            9            9       98.21%          ✓
  ...

  Test Loss     : 0.0241
  Test Accuracy : 99.18 %
```

---

## 💾 Saving & Loading the Model

The model is automatically saved after training:

```
digit_recognition_model.keras
```

To load and predict later:

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("digit_recognition_model.keras")
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = (X_test.astype("float32") / 255.0)[..., np.newaxis]

idx   = 42
probs = model.predict(X_test[idx][np.newaxis], verbose=0)[0]
pred  = np.argmax(probs)
print(f"True: {y_test[idx]}  |  Predicted: {pred}  |  Confidence: {probs[pred]*100:.2f}%")
```

---

## 📁 Project Structure

```
project2_digit_recognition.py
├── load_data()           # Download & preprocess MNIST
├── build_model()         # Define CNN architecture
├── train_model()         # Fit with early stopping
├── evaluate_model()      # Test set accuracy & loss
├── predict_samples()     # Random sample predictions
└── main()                # Orchestrates all steps
```

---

## 👤 Author

**Faizan Ansari** — SoftGrowTech Internship
