# **Facial Emotion Recognition using Fine-Tuned MobileNetV2**

## **Overview**
This project implements a **Facial Emotion Recognition (FER)** system using a fine-tuned **MobileNetV2** model. The system classifies facial expressions into seven categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

The project leverages the **FER-2013 dataset**, which contains grayscale facial images, and integrates real-time emotion recognition using a webcam.

---

## **Features**
- Utilizes a **fine-tuned MobileNetV2 model** for lightweight and efficient emotion classification.
- Incorporates **data augmentation** to improve generalization on noisy, real-world data.
- Real-time emotion recognition using a webcam with live predictions.
- Performance evaluation with accuracy, confusion matrix, and classification report.

---

## **Dataset**
### **FER-2013 Dataset**
- Source: [FER-2013 on Kaggle](https://www.kaggle.com/msambare/fer2013)
- Structure:
  ```
  dataset/
      train/
          angry/
          disgust/
          fear/
          happy/
          neutral/
          sad/
          surprise/
      test/
          angry/
          disgust/
          fear/
          happy/
          neutral/
          sad/
          surprise/
  ```

---

## **Technologies Used**
- **Python**: Core programming language.
- **TensorFlow/Keras**: Framework for building and fine-tuning the MobileNetV2 model.
- **OpenCV**: For real-time webcam integration.
- **Google Colab**: Platform for training and evaluating the model.

---

## **Model Architecture**
### **Fine-Tuned MobileNetV2**
- **Base Model**:
  - MobileNetV2 pretrained on ImageNet is used as the base.
  - All layers are unfrozen to enable fine-tuning.
- **Custom Classification Head**:
  - Global average pooling.
  - Two dense layers with dropout for regularization.
  - Output layer with softmax activation for multi-class classification.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/username/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### **2. Install Dependencies**
Use the `requirements.txt` file to install all necessary packages:
```bash
pip install -r requirements.txt
```

### **3. Dataset Preparation**
1. Download the FER-2013 dataset from Kaggle.
2. Extract the dataset into a folder named `dataset/` with the structure mentioned above.

### **4. Train the Model**
To train the model on the FER-2013 dataset, run:
```bash
python train_model.py
```

### **5. Real-Time Emotion Recognition**
Run the webcam integration script for real-time emotion detection:
```bash
python real_time_emotion.py
```

---

## **Results and Performance**
### **Model Metrics**
- **Validation Accuracy**: ~61.8% on FER-2013.
- **Test Accuracy**: ~61.7%.
- Improved performance through fine-tuning and data augmentation.

### **Confusion Matrix**
The model shows strong performance in detecting emotions like `happy` and `surprise` but struggles with `disgust` due to class imbalance in the FER-2013 dataset.

### **Visualizations**
1. Training and validation accuracy/loss curves.
2. Classification report with precision, recall, and F1 scores.
3. Confusion matrix for detailed evaluation.

---

## **Usage**
### **Training**
To train the fine-tuned MobileNetV2 model:
```bash
python train_model.py
```

### **Real-Time Emotion Detection**
- Ensure OpenCV is installed.
- Run the script to start webcam-based emotion detection:
  ```bash
  python real_time_emotion.py
  ```

The system will display the detected emotion in real time.

---

## **Future Improvements**
1. **Dataset Quality**:
   - Replace FER-2013 with higher-resolution datasets like **AffectNet** or **RAF-DB** for better accuracy.
2. **Model Enhancement**:
   - Explore deeper architectures or attention mechanisms like **CBAM** or **Vision Transformers (ViTs)**.
3. **Deployment**:
   - Optimize the model for mobile or edge devices using frameworks like TensorFlow Lite.

---

## **References**
1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
2. FER-2013 Dataset: [Kaggle Link](https://www.kaggle.com/msambare/fer2013)

---

## **Contributing**
Contributions are welcome! Feel free to fork this repository and submit a pull request with your enhancements.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.
