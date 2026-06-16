# ============================================================
#   PROJECT 2 – Handwritten Digit Recognition
#   SoftGrowTech Internship | Task 2
#   Uses: TensorFlow/Keras + MNIST dataset
# ============================================================

import numpy as np
import os

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
#  Step 1 – Load & Preprocess MNIST Dataset
# ─────────────────────────────────────────────
def load_data():
    """
    Downloads MNIST (if not cached) and normalises pixel values to [0, 1].
    Returns train/test splits as numpy arrays.
    """
    print("\n[1/4] Loading MNIST dataset …")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalise: pixel values 0-255  →  0.0-1.0
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Add channel dimension: (N, 28, 28) → (N, 28, 28, 1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    print(f"   Train samples : {X_train.shape[0]}")
    print(f"   Test  samples : {X_test.shape[0]}")
    print(f"   Image shape   : {X_train.shape[1:]}")
    return (X_train, y_train), (X_test, y_test)


# ─────────────────────────────────────────────
#  Step 2 – Build CNN Model
# ─────────────────────────────────────────────
def build_model() -> keras.Model:
    """
    Builds a Convolutional Neural Network (CNN):
      Conv → Pool → Conv → Pool → Flatten → Dense → Output
    """
    print("\n[2/4] Building CNN model …")
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),

        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),

        # Classifier head
        layers.Flatten(name='flatten'),
        layers.Dropout(0.4, name='dropout'),          # Regularisation
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dense(10,  activation='softmax', name='output'),  # 10 digit classes
    ], name='DigitRecognizer_CNN')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
#  Step 3 – Train the Model
# ─────────────────────────────────────────────
def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray) -> keras.callbacks.History:
    """
    Trains the model for up to 10 epochs with early stopping.
    Saves the best weights automatically.
    """
    print("\n[3/4] Training model …")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,   # 10 % of train used for validation
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
#  Step 4 – Evaluate & Predict
# ─────────────────────────────────────────────
def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> None:
    """Prints test-set loss and accuracy."""
    print("\n[4/4] Evaluating on test set …")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy * 100:.2f} %")
    print(f"{'=' * 40}\n")


def predict_samples(model: keras.Model,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    n: int = 10) -> None:
    """
    Runs predictions on n random test images and prints a comparison table.
    """
    print(f"  Predicting {n} random test samples:\n")
    print(f"  {'Index':>6}  {'True Label':>10}  {'Predicted':>10}  {'Confidence':>11}  {'Correct?':>8}")
    print("  " + "-" * 55)

    indices = np.random.choice(len(X_test), n, replace=False)
    correct = 0

    for idx in indices:
        img   = X_test[idx][np.newaxis, ...]        # shape (1, 28, 28, 1)
        probs = model.predict(img, verbose=0)[0]    # shape (10,)
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred])
        true  = int(y_test[idx])
        mark  = "✓" if pred == true else "✗"
        if pred == true:
            correct += 1
        print(f"  {idx:>6}  {true:>10}  {pred:>10}  {conf * 100:>10.2f}%  {mark:>8}")

    print(f"\n  Accuracy on these {n} samples: {correct}/{n}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("★" * 55)
    print("  HANDWRITTEN DIGIT RECOGNITION  –  SoftGrowTech Task 2")
    print("★" * 55)

    # 1. Data
    (X_train, y_train), (X_test, y_test) = load_data()

    # 2. Model
    model = build_model()

    # 3. Train
    train_model(model, X_train, y_train)

    # 4. Evaluate
    evaluate_model(model, X_test, y_test)

    # 5. Sample predictions
    predict_samples(model, X_test, y_test, n=10)

    # Optional: save model
    model.save("digit_recognition_model.keras")
    print("  Model saved to → digit_recognition_model.keras")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
#  BONUS: Load saved model & predict one image
# ─────────────────────────────────────────────
# Uncomment and run separately after training:
#
# import numpy as np
# import tensorflow as tf
#
# model = tf.keras.models.load_model("digit_recognition_model.keras")
# (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_test = (X_test.astype("float32") / 255.0)[..., np.newaxis]
#
# idx   = 42
# probs = model.predict(X_test[idx][np.newaxis], verbose=0)[0]
# pred  = np.argmax(probs)
# print(f"True: {y_test[idx]}  |  Predicted: {pred}  |  Confidence: {probs[pred]*100:.2f}%")
