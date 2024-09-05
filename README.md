# AI-Enhanced Medical Imaging And Diagnostic Platform

## Overview
The AI-Enhanced Medical Imaging and Diagnostic Platform is a cutting-edge solution designed to improve medical diagnostics and treatment recommendations using AI technologies. This platform integrates multiple types of inputs, such as text, audio, live recordings, and medical images, for a thorough analysis, offering precise recommendations for medications, safety precautions, and follow-up care.

The platform automates various processes such as data processing, image analysis, and text conversion to boost efficiency and reduce manual effort, while providing advanced diagnostic capabilities, even for remote areas with limited access to healthcare.

## Objective
- **Multi-Modal Submission**: Enable users to submit symptoms through text, audio, live recordings, or images for comprehensive analysis.
- **AI-Powered Disease Identification**: Utilize machine learning models and natural language processing (NLP) to provide accurate predictions and insights into potential health conditions based on user inputs.
- **User-Friendly Interface**: Develop a responsive and accessible front-end to ensure ease of use, primarily within the local development environment, enhancing the overall user experience.

## Features

### 1. Input Processing
- **Text Input**: The text is cleaned using NLP techniques (e.g., removing stopwords, tokenization) and translated to English if needed.
- **Audio Input**: Audio is recorded or uploaded as a file, converted to text using speech recognition, and processed similarly to text input.
- **Image Input**: Images are processed using a custom-trained model (`pneumonia_detection_model.h5`) to detect specific conditions like pneumonia, then cross-referenced with the dataset for further details.
- **Live Recording**: For audio, live recordings capture real-time data and save it as a file, which is then converted to text using speech recognition and processed similarly to text input.

### 2. Symptom Matching and Disease Identification
- The cleaned and translated text is vectorized using **TfidfVectorizer**, and cosine similarity is calculated between the user input and the dataset's symptoms.
- The system identifies the most similar disease based on this similarity score.

### 3. Disease Information Retrieval
- Once a disease is identified, detailed information (e.g., symptoms, treatment plan, recommended medications) is retrieved from the combined dataset.
- If needed, this information is translated back to the user's preferred language.

### 4. Output
- Relevant disease information is formatted and printed clearly, ensuring all critical details are covered.
- If no close match is found, the system prompts the user to try different symptoms.

## Technologies Used
- **Natural Language Processing (NLP)**
- **Audio Processing**
- **Image Processing and Computer Vision**
- **Machine Learning**
- **Data Handling**


