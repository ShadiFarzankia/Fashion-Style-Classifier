
# Fashion Style Classifier

A comprehensive workflow for classifying fashion images into style categories using both deep learning (CNN) and classical machine learning (KNN). This project demonstrates data collection, preprocessing, model training, evaluation, and reporting for fashion style recognition.

---

## Experimental Setup & Key Findings

### Experimental Setup
- **Data Source:** Images scraped from Reddit (r/streetwear, r/fashion, r/femalefashionadvice), manually labeled using Label Studio.
- **Preprocessing:** Cleaning missing/corrupt files, class balance visualization, stratified train/test split.
- **Models:**
  - **CNN:** EfficientNet-B0 (PyTorch, timm), transfer learning, data augmentation, 20 epochs, Adam optimizer.
  - **KNN:** Deep features from MobileNetV2 (Keras), PCA for dimensionality reduction, KNeighborsClassifier (scikit-learn), GridSearchCV for hyperparameter tuning.
- **Evaluation:** Accuracy, F1-score, confusion matrix, classification report.
- **Reproducibility:** All code and steps are in Jupyter notebooks with cell-by-cell documentation.

### Key Findings
- Both CNN and KNN achieved ~41% accuracy, with best F1-scores for Formal and Streetwear classes (~0.55).
- Sporty class was poorly classified (F1 ~0.00) due to severe class imbalance.
- Data augmentation and transfer learning improved CNN performance, but minority classes remain challenging.
- KNN with deep features is a strong baseline but less robust than CNN for complex patterns.

---

## Challenges & Literature Context

### Challenges
- **Class Imbalance:** Sporty and Vintage classes had far fewer samples, leading to poor generalization and low F1-scores.
- **Noisy Real-World Data:** Images scraped from Reddit are diverse, with varying quality, backgrounds, and styles, making feature extraction and classification difficult.
- **Limited Labeled Data:** Manual annotation is time-consuming, restricting dataset size and diversity.
- **Model Selection:** Balancing between deep learning (CNN) and classical ML (KNN) for a non-standard dataset.

### Literature Context
- **Existing Solutions:**
  - Deep learning models (EfficientNet, ResNet, Vision Transformers) are state-of-the-art for image classification ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)).
  - Transfer learning and data augmentation are widely used to improve performance on small datasets.
  - Class imbalance is a common issue; solutions include oversampling, class weighting, and focal loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)).
  - KNN with deep features is a simple, interpretable baseline but less effective for complex, imbalanced data.

---

## Repository Navigation & Rerunning Experiments

### Repo Structure
- `data/images/` — Fashion images
- `data/labels/labels_file.csv` — CSV with image filenames and style labels
- `fashion_image_classification_CNN.ipynb` — CNN workflow (EfficientNet-B0)
- `fashion_image_classification_knn.ipynb` — KNN workflow (MobileNetV2 features)
- `requirements.txt` — Python dependencies

### How to Rerun Experiments
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AadrianLeo/Fashion-Style-Classifier.git
   cd Fashion-Style-Classifier
   ```
2. **Install dependencies:**
   - Python 3.8+
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - For KNN feature extraction:
     ```bash
     pip install tensorflow keras
     ```
   - For CNN:
     ```bash
     pip install torch torchvision timm
     ```
3. **Prepare the data:**
   - Place the labeled CSV and image files in the expected directories (`data/images/`, `data/labels/labels_file.csv`).
4. **Run notebooks:**
   - Open `fashion_image_classification_CNN.ipynb` and `fashion_image_classification_knn.ipynb` in Jupyter or VS Code.
   - Run all cells sequentially; each cell is documented for clarity.
   - Results (metrics, confusion matrices, sample predictions) are displayed in the output cells.

---

## Table of Contents
- Project Overview
- Dataset
- Workflow
- Models
  - CNN (EfficientNet-B0)
  - KNN
- Results
- Future Work
- How to Run
- References

---

## Project Overview
This project aims to classify fashion images into five style categories: **casual**, **formal**, **streetwear**, **sporty**, and **vintage**. Images were scraped from Reddit and manually labeled using Label Studio. The workflow includes feature extraction, data preprocessing, model training (CNN & KNN), evaluation, and visualization.

## Dataset
- **Source:** Reddit (subreddits: r/streetwear, r/fashion, r/femalefashionadvice)
- **Annotation:** Manual labeling via Label Studio
- **Structure:**
  - `data/images/` — Fashion images
  - `data/labels/labels_file.csv` — CSV with image filenames and style labels

## Workflow
1. **Data Collection:** Scrape images from Reddit using PRAW and requests.
2. **Annotation:** Label images in Label Studio; export CSV.
3. **Preprocessing:**
   - Clean missing labels
   - Visualize samples and class distribution
   - Stratified train/test split
4. **Feature Extraction:**
   - CNN: Use raw images
   - KNN: Extract features using MobileNetV2
5. **Model Training:**
   - CNN: EfficientNet-B0 (PyTorch, timm)
   - KNN: Scikit-learn pipeline (scaling, PCA, KNN)
6. **Evaluation:**
   - Accuracy, F1-score, confusion matrix
   - Visualizations and result summaries

## Models

### CNN (EfficientNet-B0)
- **Framework:** PyTorch, timm
- **Architecture:** EfficientNet-B0, pretrained on ImageNet, fine-tuned for 5 classes
- **Augmentation:** Resize, random flip, rotation, color jitter
- **Training:** 20 epochs, Adam optimizer, cross-entropy loss
- **Evaluation:** Classification report, confusion matrix

### KNN
- **Feature Extraction:** MobileNetV2 (Keras, TensorFlow)
- **Dimensionality Reduction:** PCA (retain 95% variance)
- **Classifier:** KNeighborsClassifier (scikit-learn)
- **Hyperparameter Tuning:** GridSearchCV (n_neighbors, weights, metric)
- **Evaluation:** Accuracy, F1-score, confusion matrix

## Results
- **Overall Accuracy:** ~41% (both CNN & KNN)
- **Best Classified Styles:** Formal, Streetwear (F1 ~0.55)
- **Worst Classified Style:** Sporty (F1 ~0.00, due to class imbalance)
- **Macro Avg. F1-score:** 0.29
- **Confusion Matrix:** See notebook for visualizations

## Future Work
- Explore advanced architectures (EfficientNetV2, Vision Transformers)
- Apply more aggressive data augmentation
- Address class imbalance with oversampling or synthetic data
- Fine-tune hyperparameters and try alternative feature extractors
- Integrate explainability tools (e.g., Grad-CAM)
- Deploy best model as a web/mobile app

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AadrianLeo/Fashion-Style-Classifier.git
   cd Fashion-Style-Classifier
   ```
2. **Install dependencies:**
   - Python 3.8+
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - For KNN feature extraction:
     ```bash
     pip install tensorflow keras
     ```
   - For CNN:
     ```bash
     pip install torch torchvision timm
     ```
3. **Run notebooks:**
   - Open `fashion_image_classification_CNN.ipynb` and `fashion_image_classification_knn.ipynb` in Jupyter or VS Code
   - Follow cell-by-cell instructions

## References
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Label Studio](https://labelstud.io/)
- [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/)

---

For questions or contributions, please open an issue or submit a pull request.