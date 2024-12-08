### **Emotion Recognition Using Fine-Tuned MobileNetV2**

---

#### **Overview**
This project implements a Fine-Tuned MobileNetV2 model for emotion recognition using the FER-2013 dataset. The objective is to classify facial expressions into seven emotional categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. The MobileNetV2 architecture was fine-tuned for optimal performance, leveraging its lightweight design and efficient depth-wise separable convolutions to achieve computational efficiency without compromising accuracy.

---

#### **Dataset**
- **Name:** FER-2013 (Facial Expression Recognition 2013)
- **Description:** Contains 35,887 grayscale facial images (48x48 pixels) divided into:
  - **Training Set:** 28,709 images
  - **Test Set:** 7,178 images
- **Challenges:**
  - Low resolution
  - Class imbalance
  - Overlapping emotional features

---

#### **Model Details**
- **Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Modifications:**
  - GlobalAveragePooling2D layer for feature extraction
  - Dense layer with 128 neurons (ReLU activation)
  - Dropout layers (0.5) for regularization
  - Output layer with 7 neurons (softmax activation) for emotion classification
- **Hyperparameters:**
  - **Learning Rate:** `1e-5` with a learning rate scheduler
  - **Batch Size:** `32`
  - **Epochs:** `30`
- **Preprocessing:**
  - Normalization (rescaling pixel values to `[0, 1]`)
  - Data augmentation (rotation, zoom, flip, etc.) to increase variability and robustness

---

#### **Performance**
- **Test Accuracy:** 50.54%
- **Validation Loss:** 1.341
- **Strengths:**
  - Lightweight architecture
  - Optimized for FER-2013's constraints
- **Weaknesses:**
  - Limited by dataset quality (low resolution and imbalance)

---

#### **How to Use**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/fine-tuned-mobilenetv2-emotion-recognition.git
   cd fine-tuned-mobilenetv2-emotion-recognition
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   Update dataset paths in the code and execute the training script:
   ```bash
   python train.py
   ```

4. **Evaluate the Model:**
   Run the evaluation script to generate metrics:
   ```bash
   python evaluate.py
   ```

5. **Test on New Images:**
   Use the inference script to test the model on new images:
   ```bash
   python inference.py --image_path <path_to_image>
   ```

---

#### **Project Structure**
```plaintext
|-- dataset/                    # FER-2013 Dataset (train/test)
|-- models/                     # Saved models
|   |-- mobilenetv2_fine_tuned_best.keras
|-- src/                        # Source files
|   |-- train.py                # Training script
|   |-- evaluate.py             # Evaluation script
|   |-- inference.py            # Inference script
|-- results/                    # Performance metrics and visualizations
|   |-- classification_report.txt
|   |-- confusion_matrix.png
|   |-- training_history.png
|-- README.md                   # Project documentation
|-- requirements.txt            # Python dependencies
```

---

#### **Results**
- **Classification Report:** Precision, recall, and F1-score for all classes.
- **Confusion Matrix:** Visualizes misclassifications across emotional categories.
- **Training History:** Accuracy and loss curves for training and validation.

---

#### **Future Work**
- Transition to higher-quality datasets like AffectNet for improved resolution and diversity.
- Experiment with advanced architectures such as Vision Transformers or hybrid models.
- Address class imbalance using focal loss or synthetic data generation.

---

#### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

#### **Acknowledgments**
- FER-2013 dataset contributors
- TensorFlow and Keras open-source communities

---
