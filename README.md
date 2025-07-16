# Fashion-Style-Classifier

## Project Overview
Fashion Style Classifier is an end-to-end machine learning project for classifying fashion images into style categories using a real-world, non-standard dataset. The project demonstrates robust data preprocessing, model training, error handling, and model comparison.

## Features
- Data curation, cleaning, and preprocessing
- Exploratory data analysis and class balance visualization
- Robust error handling for missing/corrupt files
- Model training and evaluation:
  - Deep learning (CNN with MobileNetV2, transfer learning, fine-tuning, augmentation, class weights)
  - Classical machine learning (KNN on deep features)
  - Naive baseline (most frequent class)
- Model comparison using accuracy, precision, recall, and F1-score
- K-fold cross-validation for KNN
- Reproducible code and clear workflow

## Getting Started
1. **Clone the repository:**
   ```
   git clone https://github.com/AadrianLeo/Fashion-Style-Classifier.git
   ```
2. **Install dependencies:**
   - Create a virtual environment (recommended)
   - Install packages from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
3. **Prepare the data:**
   - Place the labeled CSV and image files in the expected directories (see notebook for details).
4. **Run the notebook:**
   - Open `Fashion_style_classifier.ipynb` in Jupyter or VS Code and run all cells top-to-bottom.

## Usage
- The notebook is robust to missing files and errors.
- Key metrics, confusion matrices, and sample predictions are displayed for each model.
- Results for CNN, KNN, and the naive baseline are reported and compared.

## Model Comparison
| Model            | Strengths                                              | Weaknesses                                         | Best Use Case                                 |
|------------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------|
| CNN (MobileNetV2)| High accuracy, learns complex patterns, robust to noise| Requires more data, longer training time           | Large, diverse datasets; production systems   |
| KNN              | Simple, interpretable, works with deep features       | Slower on large datasets, less robust to noise     | Quick baselines, small/medium datasets        |
| Naive Baseline   | Fast, easy to implement                              | Ignores input features, very low accuracy         | Reference for minimum expected performance    |

## Results Summary
Fill in your actual results after running the notebook. Example:
- **CNN (MobileNetV2):** Accuracy = 0.85, Precision = 0.84, Recall = 0.83, F1-score = 0.83
- **KNN:** Accuracy = 0.72, Precision = 0.70, Recall = 0.71, F1-score = 0.70
- **Naive Baseline:** Accuracy = 0.40, Precision = 0.16, Recall = 0.40, F1-score = 0.23

## Team & Contributions
- List team members and their contributions here.

## License
This project is for educational purposes.