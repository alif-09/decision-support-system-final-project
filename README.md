# Suicide Detection Project -- Decision Support System Final Project

This project focuses on creating a machine learning-based application to detect suicidal tendencies from text entries. By leveraging Natural Language Processing (NLP) and deep learning, the model achieves high accuracy in classifying text as either "suicide" or "non-suicide." The application supports multiple languages, enabling broader usability across diverse populations.

## Table of Contents
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [Application Components](#application-components)
- [Deployment](#deployment)
- [How to Run Locally](#how-to-run-locally)
- [Example Predictions](#example-predictions)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Team Members
- **Muhammad Zhafran Shiddiq** (140810220007)  
- **Tegar Muhamad Rizki** (140810220034)  
- **Alif Al Husaini** (140810220036)

---

## Dataset
- **Source**: [Kaggle - Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)  
- **Description**:  
  The dataset contains three columns, of which the primary columns used are:
  - `text`: The textual input provided by users.
  - `class`: The target variable indicating "suicide" or "non-suicide."

---

## Project Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Analyzed the structure of the dataset to identify data trends and patterns.
   - Visualized key statistics to understand class distribution and word usage.

2. **Data Preprocessing**  
   - **Standar Preprocessing**: Lowercasing, Removing Stopwords, Replacing Abbreviations, Word Segmentation, Removing Emojis, Removing Special Csaracters, and Text Lemmatization
   - **Tokenization**: Used TensorFlow/Keras Tokenizer to split text into individual tokens.  
   - **Label Encoding**: Converted the `class` column into binary matrices using `pd.get_dummies()`.  
   - **Data Splitting**: Divided the data into 70% for training and 30% for testing.  
   - **Word Embedding**: Employed Word2Vec or GloVe (via Gensim) to create low-dimensional vector representations of words, improving model accuracy by capturing semantic meanings.

3. **Model Building and Training**  
   - Developed a deep learning classification model using GRU with TensorFlow/Keras.  
   - Achieved optimal results with a model trained for 10 epochs.

4. **Model Evaluation**  
   - Performance metrics:
     - **Precision**: 0.94 (suicide), 0.93 (non-suicide)
     - **Recall**: 0.93 (suicide), 0.94 (non-suicide)
     - **F1-Score**: 0.93 for both classes.
   - Overall Accuracy: **93%**  

---

## Model Performance
The model achieves a **validation accuracy of 93%** with balanced precision, recall, and F1-Score metrics. This demonstrates its robustness in detecting suicidal tendencies.

---

## Application Components
The repository contains the following core components:
- **app.py**: Main script for running the Streamlit application.
- **utils.py**: Utility functions for:
  - Loading the trained model and tokenizer.
  - Preprocessing input text.
  - Generating predictions.
- **best_model.keras**: Pre-trained model in Keras format.
- **tokenizer.pkl**: Serialized tokenizer used for input text preprocessing.
- **requirements.txt**: Python dependencies required for running the application.

---

## Deployment
The project is deployed using [Streamlit](https://streamlit.io/).  
- Access the live application here: [Suicidal Detection App](https://suicidal-detection-dss-final-project.streamlit.app/).

---

## How to Run Locally
To run the application locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/alif-09/decision-support-system-final-project.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Example Predictions
The application accepts text input and predicts whether the text exhibits suicidal tendencies.  
Example outputs include:  
- **Input**: "I feel so hopeless and don’t know if I can continue."  
  - **Prediction**: Suicide  
- **Input**: "I’m looking forward to the weekend!"  
  - **Prediction**: Non-Suicide  

---

## Future Improvements
- Incorporate larger and more diverse datasets to enhance generalizability.  
- Explore transformer-based models like BERT or GPT for improved performance.  
- Add a real-time alert system to notify mental health professionals.  
- Further enhance multilingual support to improve accuracy in non-English languages.

---

## Acknowledgements
This project would not have been possible without the following resources:
- [Kaggle - Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)  
- TensorFlow/Keras for building the machine learning model.  
- Gensim for word embedding.

