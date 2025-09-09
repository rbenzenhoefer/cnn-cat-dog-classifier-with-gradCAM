# CNN Cat vs Dog Classifier

Deep learning project implementing a Convolutional Neural Network for binary classification of cat and dog images, featuring Grad-CAM visualization for model interpretability.

## Results
- **Test Accuracy**: 90.6%
- **Precision**: 100.0% 
- **Recall**: 82.4%
- **Model Size**: 3.7M parameters

## Features
- Functional Keras CNN architecture
- Grad-CAM implementation for visual explanations
- Comprehensive data preprocessing pipeline
- Domain gap analysis and model interpretability

## Quick Start
```bash
git clone https://github.com/rbenzenhoefer/cnn-cat-dog-classifier-with-gradCAM.git
cd cnn-cat-dog-classifier
pip install -r requirements.txt
python main.py
```

## Data
The model was trained on 143 manually curated images (80 cats, 63 dogs) 
collected from Google Images. Due to size constraints, the dataset is not 
included in this repository. 'fix_filenames' can be used to get rid of
problematic characters in filenames.

## Trained Model
The trained model achieves 90.6% test accuracy. To reproduce results, 
run `python main.py` with your own dataset.

## Files
- `main.py` - Model training and evaluation
- `gradcam_functional.py` - Grad-CAM visualization
- `fix_filenames.py` - Data preprocessing utilities
- `Technical Documentation (2).pdf` - Complete technical documentation

## Key Learnings
- Domain gap challenges in real-world deployment
- Importance of data quality over model complexity
- Model interpretability through gradient-based methods

## Author
Raphael Benzenhöfer