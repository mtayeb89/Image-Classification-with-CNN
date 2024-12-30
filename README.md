# CNN Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The implementation provides a flexible and reusable structure for training image classification models on custom datasets.

## Features
- Customizable CNN architecture
- Built-in data preprocessing
- Support for custom datasets
- Model saving and loading
- Evaluation metrics
- Batch processing

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## Project Structure
```
image_classifier/
├── image_classifier.py    # Main implementation
├── README.md             # Documentation
└── data/                 # Dataset directory
    ├── class1/          
    ├── class2/
    └── ...
```

## Usage
1. **Basic Usage**
```python
from image_classifier import ImageClassifier

# Initialize classifier
classifier = ImageClassifier(input_shape=(28, 28, 1), num_classes=10)

# Load and preprocess data
X, y = classifier.load_and_preprocess_data('path/to/data')

# Train the model
history = classifier.train(X, y, epochs=10)

# Save the model
classifier.save_model('model.h5')
```

2. **Loading a Saved Model**
```python
classifier = ImageClassifier()
classifier.load_model('model.h5')
```

3. **Making Predictions**
```python
prediction = classifier.predict(image)
```

## Data Format
The dataset should be organized in the following structure:
```
data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Model Architecture
The CNN architecture consists of:
- 3 Convolutional layers with ReLU activation
- Batch Normalization layers
- MaxPooling layers
- Dense layers with dropout
- Softmax output layer

## Training Parameters
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Default batch size: 32
- Default epochs: 10

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
