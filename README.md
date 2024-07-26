# Pill Detection and Medical Chatbot for Disease Prediction

## Project Overview

This project presents a comprehensive system for healthcare that integrates pill detection and disease prediction through a medical chatbot. The system comprises two main components:
1. **Pill Detection**: Utilizes a Convolutional Neural Network (CNN) to analyze images of pills and predict their type.
2. **Disease Prediction**: Employs a Decision Tree classifier to predict potential diseases based on user-reported symptoms.

The proposed system aims to improve medication adherence and provide early disease detection, enhancing the overall healthcare experience.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Features
- **Pill Detection**: Classifies pills from images with an accuracy of 81%.
- **Disease Prediction**: Predicts diseases based on symptoms with an accuracy of 91%.
- **User-Friendly Interface**: Web application interface for easy interaction.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pill-detection-medical-chatbot.git
    cd pill-detection-medical-chatbot
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the datasets and place them in the `data` directory as specified in the project.

## Usage

1. **Run the Flask Application**:
    ```bash
    python app.py
    ```

2. Open a web browser and navigate to `http://localhost:5000` to access the web application.

3. **Pill Detection**:
    - Upload an image of a pill.
    - The system will predict the pill type and display the result.

4. **Disease Prediction**:
    - Enter symptoms into the chatbot.
    - The system will predict the disease and provide a health report.

## Data Collection

- **Pill Dataset**: Contains 500 images for each of 10 pill classes.
- **Disease Dataset**: Kaggle dataset with information on 40 diseases and their symptoms.

## Architecture

The system consists of two main components:
1. **Pill Detection**: A CNN model trained for image classification.
2. **Disease Prediction**: A Decision Tree classifier for disease prediction based on symptoms.

Refer to the architecture diagram below:

![System Architecture](architecture_diagram.png)

## Methodology

### Pill Detection
- Uses TensorFlow and Keras for building and training the CNN model.
- Processes input images and predicts pill types with confidence scores.

### Disease Prediction
- Uses pandas and numpy for data handling.
- Trains a Decision Tree Classifier and evaluates symptom-based disease predictions.

## Results

- **Pill Detection Accuracy**: 81%
- **Disease Prediction Accuracy**: 91%

Comparison of model performances is shown below:

![CNN vs RNN](table1.png)

## Future Work

- Integrate a feature for scheduling doctor appointments.
- Expand the disease prediction model to include more diseases.
- Enhance pill detection for better accuracy with low-resolution images.

## Acknowledgements

- **Dr. Priyanka H** for guidance and support.
- **PES University** for providing the resources for this project.

## Contact

For any questions or suggestions, please contact [Sudeepa N S](mailto:youremail@example.com).

