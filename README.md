# Dog Breed Classification with Transfer Learning

This project implements an end-to-end deep learning pipeline for **multi-class image classification** using **TensorFlow 2 and Keras**. The goal is to classify images of dogs into **120 different breeds** using **transfer learning** with a pretrained **MobileNetV2** model.

The project is designed and trained in **Google Colab** and follows best practices for data preprocessing, model training, evaluation, and inference.

---

## Project Overview

- **Task:** Multi-class image classification  
- **Classes:** 120 dog breeds  
- **Dataset:** Kaggle Dog Breed Identification  
- **Framework:** TensorFlow 2 / Keras  
- **Model:** MobileNetV2 (transfer learning)  

The pretrained model is used as a feature extractor and fine-tuned for the dog breed classification task, enabling efficient training with strong performance.

---

## Key Concepts Demonstrated

- Transfer learning with pretrained CNNs
- Image preprocessing and batching with `tf.data`
- Handling one-hot encoded labels
- Overfitting detection and mitigation
- Model evaluation and inference
- Saving and loading models using the native Keras format

___


> **Note:** The dataset and trained models are not committed to GitHub due to size constraints.

---

## Data Pipeline

- Images are loaded from disk using file paths
- Images are resized to **224 × 224**
- Pixel values are scaled to **[0, 1]**
- Labels are one-hot encoded (`float32`)
- Batching and shuffling are handled via `tf.data.Dataset`

---

## Model Architecture

- **Base model:** MobileNetV2 (ImageNet weights, frozen)
- **Head:**
  - Global Average Pooling
  - Dropout for regularization
  - Dense softmax layer (120 classes)

The input images are normalized to **[-1, 1]** to match MobileNetV2 expectations.

---

## Training

- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metrics: Accuracy  
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - TensorBoard logging

---

## Results

The model achieves strong training performance and reasonable validation accuracy given the number of classes and dataset size. Overfitting is mitigated through regularization and early stopping.

---

## Inference

The trained model can be used to:
- Predict dog breeds on unseen images
- Generate class probabilities
- Visualize predictions
- (Optionally) create Kaggle-style submission files

---

## Model Persistence

Models are saved using the **native Keras format**:

```python
model.save("dog_breed_mobilenetv2.keras")


