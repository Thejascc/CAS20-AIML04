# CAS20-AIML04
testing
# Cyberbullying Detection Using Logistic Regression

## Problem Statement
Cyberbullying has become a major concern on social media platforms, leading to serious psychological and emotional impacts. This project aims to detect cyberbullying in tweets using Natural Language Processing (NLP) techniques and a Logistic Regression model. The dataset consists of tweets labeled with different types of cyberbullying, and the objective is to classify them accurately.

## Tech Stack Used
- **Python**
- **Pandas & NumPy**
- **Natural Language Processing (NLTK)**
- **Scikit-learn (Machine Learning)**
- **TF-IDF Vectorization**
- **Logistic Regression**

## Dataset
The dataset used for this project is `cyberbullying_tweets.csv`, which contains tweets and their corresponding cyberbullying labels.

## Project Workflow

### Data Preprocessing
- Cleaning text by removing special characters, converting to lowercase, and removing stopwords.
- Lemmatization to normalize words.

### Feature Extraction
- Converting text data into numerical form using **TF-IDF Vectorization**.

### Model Training
- Splitting the dataset into training and testing sets.
- Training a **Logistic Regression** model.

### Model Evaluation
- Predicting cyberbullying categories for unseen tweets.
- Evaluating model performance using **accuracy score** and **classification report**.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python main.py
   ```

## Results
- The model classifies tweets into different cyberbullying categories with good accuracy.
- Performance metrics such as precision, recall, and F1-score are used to evaluate the model.

## Future Enhancements
- **Deep Learning Models:** Implement deep learning techniques like LSTMs or transformers (BERT) to improve classification accuracy.
- **Data Augmentation:** Expand the dataset with more labeled tweets to enhance model performance.
- **Real-time Detection:** Develop a real-time cyberbullying detection system for social media monitoring.
- **Multi-class Classification:** Improve handling of multiple cyberbullying categories for better granularity.
- **Explainability & Interpretability:** Integrate SHAP or LIME to interpret model predictions.

## Contributing
Feel free to open issues or submit pull requests for improvements!

## License
This project is licensed under the MIT License.

