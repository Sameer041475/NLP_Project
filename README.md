# 📄 Resume Category Prediction using NLP & Machine Learning

An NLP-based machine learning application that automatically predicts the **job category of a resume** from its text.

The project uses **TF-IDF Vectorization** to convert resume text into numerical features and **Linear Support Vector Classification (LinearSVC)** to classify resumes into different job categories.

A **Streamlit web application** is included for real-time resume category prediction.

---

## 🚀 Project Overview

Recruiters often receive a large number of resumes for different job roles. Manually categorizing these resumes can be time-consuming.

This project automates the initial resume classification process:

```text
Resume Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
LinearSVC Model
     ↓
Predicted Job Category
```

The trained model can take new resume text as input and predict the most suitable category based on patterns learned from the training dataset.

---

## ✨ Features

* 📄 Resume text classification
* 🔤 Natural Language Processing (NLP)
* 📊 TF-IDF feature extraction
* 🤖 LinearSVC machine learning model
* ⚙️ Hyperparameter tuning using different `C` values
* 💾 Saved trained model using Pickle
* 💾 Saved TF-IDF vectorizer
* 🌐 Interactive Streamlit web application
* ⚡ Real-time prediction for new resume text

---

## 🛠️ Technologies Used

| Technology       | Purpose                  |
| ---------------- | ------------------------ |
| Python           | Programming language     |
| Pandas           | Dataset handling         |
| Scikit-learn     | Machine learning and NLP |
| TF-IDF           | Text feature extraction  |
| LinearSVC        | Resume classification    |
| Pickle           | Model serialization      |
| Streamlit        | Web application          |
| Jupyter Notebook | Model development        |

---

## 📂 Project Structure

```text
NLP_Project/
│
├── app.py                    # Streamlit application
├── Resume.csv                # Resume dataset
├── Untitled.ipynb            # Model development notebook
├── svm_model.pkl             # Trained LinearSVC model
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── .gitignore                # Git ignored files
└── README.md                 # Project documentation
```

---

## 🔄 Machine Learning Pipeline

### 1. Dataset

The project uses a resume dataset containing:

* Resume text
* Resume category/target label

The main text column used for NLP is:

```python
df["Resume_str"]
```

The target column is:

```python
df["Category"]
```

---

### 2. Train-Test Split

The dataset is divided into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["Resume_str"],
    df["Category"],
    test_size=0.2,
    random_state=42,
    stratify=df["Category"]
)
```

---

### 3. TF-IDF Vectorization

Resume text is converted into numerical features using TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

The model uses:

* **Unigrams** — individual words
* **Bigrams** — two-word combinations
* English stop-word removal
* Minimum document frequency filtering
* Maximum document frequency filtering
* Sublinear term frequency

---

## 🤖 Machine Learning Model

The project uses:

```python
from sklearn.svm import LinearSVC

clf = LinearSVC(
    C=0.7,
    max_iter=5000
)

clf.fit(X_train, y_train)
```

Different values of `C` were tested to find a suitable model configuration.

Example:

```text
C = 0.3 → 66.60%
C = 0.4 → 67.00%
C = 0.5 → 67.81%
C = 0.6 → 67.61%
C = 0.7 → 68.01%
C = 0.8 → 67.81%
C = 0.9 → 67.20%
C = 1.0 → 67.61%
```

The tested configuration with `C = 0.7` achieved approximately **68% accuracy** on the held-out test set in this experiment.

> Accuracy can vary depending on the dataset, preprocessing, train-test split, and environment.

---

## 💾 Saving the Model

The trained model and fitted TF-IDF vectorizer are saved using Pickle:

```python
import pickle

with open("svm_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
```

Both files are required for prediction.

---

## 🌐 Streamlit Application

The project includes a Streamlit interface where users can enter resume text and receive a predicted category.

### Application Flow

```text
User enters resume
        ↓
TF-IDF Vectorizer
        ↓
Numerical feature representation
        ↓
LinearSVC Model
        ↓
Predicted Category
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sameer041475/NLP_Project.git
```

### 2. Open the project

```bash
cd NLP_Project
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn streamlit
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Using the Application

1. Start the Streamlit application.
2. Enter resume information into the text box.
3. Click **Predict Category**.
4. The trained LinearSVC model processes the resume.
5. The predicted job category is displayed.

Example input:

```text
Python, Machine Learning, TensorFlow, Pandas,
NumPy, SQL, Data Analysis, Scikit-learn
```

The application then returns the predicted category.

---

## 📌 Important Files

### `app.py`

Contains the Streamlit user interface and prediction logic.

### `svm_model.pkl`

Contains the trained LinearSVC model.

### `tfidf_vectorizer.pkl`

Contains the fitted TF-IDF vectorizer used during model training.

### `Untitled.ipynb`

Contains the data processing, TF-IDF transformation, model training, evaluation, and experimentation.

### `Resume.csv`

Contains the resume dataset used for the project.

---

## 📊 Model Architecture

```text
                 Resume Dataset
                       │
                       ▼
              Text Preprocessing
                       │
                       ▼
                TF-IDF Vectorizer
                       │
                       ▼
              Training/Test Split
                       │
                       ▼
                  LinearSVC
                       │
                       ▼
              Category Prediction
                       │
                       ▼
                Streamlit App
```

---

## 🔮 Future Improvements

* Improve text preprocessing
* Experiment with Logistic Regression and Naive Bayes
* Try advanced NLP models such as BERT
* Add resume PDF upload functionality
* Extract text automatically from PDF resumes
* Display prediction confidence or decision scores
* Add resume skill extraction
* Add a recommendation system for suitable job roles
* Deploy the Streamlit application online
* Improve model performance with larger and more diverse datasets

---

## 🎯 Learning Outcomes

Through this project, I practiced:

* Natural Language Processing
* Text preprocessing
* TF-IDF feature engineering
* Train-test splitting
* Support Vector Machines
* Hyperparameter tuning
* Model evaluation
* Model serialization using Pickle
* Building ML applications with Streamlit
* Deploying machine learning workflows into an interactive application

---

## 👨‍💻 Author

**Sameer Kampa**

CSE – Artificial Intelligence Student
Interested in **AI/ML, NLP, Generative AI, and Software Development**.

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
