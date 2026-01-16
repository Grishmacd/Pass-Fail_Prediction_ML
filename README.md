# Pass/Fail Prediction Using TF-IDF + Naive Bayes (Machine Learning)

This project builds a simple **binary classification** model to predict whether a student result is **Pass (yes)** or **Fail (no)** using a text-style feature created from the dataset. It uses **TF-IDF** to convert input into numeric features and **Multinomial Naive Bayes** for classification. The project follows the ML workflow:

**Problem Statement → Selection of Data → Collection of Data → EDA → Train/Test Split → Model Selection → Evaluation Metric**

---

## Problem Statement
Predict whether the outcome is **Pass** or **Fail** based on the available input feature.

**Input Used in Code:**
- `subject1` (converted to string and stored in `message`)

**Output:**
- Predicted label: `yes` (Pass) or `no` (Fail)

---

## Selection of Data
**Dataset Type Used:** Structured tabular dataset (CSV) with binary labels

Why this dataset is suitable:
- Clear target column (`Pass`: yes/no)
- Easy to convert into a classification task
- Good beginner project for understanding the full pipeline

---

## Collection of Data
The dataset is uploaded in Google Colab using:
- `from google.colab import files`
- `files.upload()`

Then loaded using:
- `pd.read_csv("file_pass_fail.csv")`

---

## EDA (Exploratory Data Analysis)
EDA is kept simple to ensure correct training:
- Confirm dataset is loaded correctly
- Verify `subject1` and `Pass` columns exist
- Check label mapping:
  - `yes → 1`
  - `no → 0`

---

## Dividing Training and Testing
The dataset is split using `train_test_split`:
- Training set: model learns from examples
- Testing set: model is evaluated on unseen rows

---

## Data Preprocessing (Text to Numbers)
The model cannot directly understand raw input, so the project converts input into numeric features using **TF-IDF**:

Used in code:
- `TfidfVectorizer()`
- `fit_transform` on training messages
- `transform` on test messages

---

## Model Selection
**Model used:** Multinomial Naive Bayes (`MultinomialNB`)

Why Naive Bayes:
- Fast and effective for TF-IDF style features
- Works well for simple classification tasks

---

## Evaluation Metric (Used in this Project)
This project evaluates the model using:
- **Accuracy Score** (overall correctness)

Used in code:
- `accuracy_score(y_test, y_pred)`

---

## Custom Prediction (Inference)
A helper function is created to test new inputs:
- Converts input to string
- Applies TF-IDF transform
- Predicts Pass/Fail as `yes` or `no`

Example tests in code:
- `predict_message(40)`
- `predict_message(15)`

---

## Main Libraries Used (and why)

1. `pandas`  
   - Loads the CSV dataset and manages it as a DataFrame.

2. `sklearn.model_selection.train_test_split`  
   - Splits data into train and test sets.

3. `sklearn.feature_extraction.text.TfidfVectorizer`  
   - Converts the input feature into TF-IDF numeric vectors.

4. `sklearn.naive_bayes.MultinomialNB`  
   - Trains the binary classification model.

5. `sklearn.metrics.accuracy_score`  
   - Evaluates performance using accuracy.

6. `google.colab.files`  
   - Uploads the dataset file into Colab.

---

## Output
- Printed **Accuracy**
- Predicted output for custom inputs:
  - `yes` or `no`

---

## Developer
Grishma C.D
