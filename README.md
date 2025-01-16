# IMDb Rating Classification Using Logistic Regression, Random Forest, Gradient Boosting, and MLP Classifier

## **Author**: Elif Sena Daldal  


---

## **Abstract**
This project classifies IMDb titles into **Top Rated** and **Worst Rated** categories based on average ratings, genres, and metadata. It explores and compares the performance of four machine learning models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron (MLP) Classifier

Comprehensive preprocessing techniques like handling missing values, one-hot encoding genres, and normalizing numerical features were applied. The models were evaluated using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score** to identify the best-performing classifier.

---

## **Introduction**
IMDb is a popular database providing user ratings for movies and TV shows. This project aims to classify titles into:
- **Top Rated**: Average rating ≥ 8.0
- **Worst Rated**: Average rating ≤ 4.0

### **Objectives**
1. Address missing values, transform categorical and numerical features, and preprocess data for machine learning.
2. Implement and compare Logistic Regression, Random Forest, Gradient Boosting, and MLP Classifier for binary classification.
3. Identify the best-performing model using evaluation metrics.

### **Key Findings**
- **Gradient Boosting**: Best accuracy at **81.47%**.
- **Random Forest & MLP Classifier**: Achieved comparable accuracies of **80.55%** and **80.5%**.
- **Logistic Regression**: Lowest accuracy at **78.59%**, but still efficient and interpretable.

---

## **Dataset**
The dataset includes IMDb metadata such as:

- **Source**: [Full IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/octopusteam/full-imdb-dataset)
The dataset includes IMDb metadata such as:
- Average ratings
- Genres
- Release years
- Number of votes

**Preprocessing Steps**:
1. Handled missing values:
   - Median imputation for numerical columns (e.g., `numVotes` and `releaseYear`).
   - Filled missing genres with `"Unknown"`.
2. One-hot encoding for genres and types (e.g., movie, tvSeries).
3. Min-Max normalization for numerical features.
4. Created binary labels:
   - `1` for **Top Rated**.
   - `0` for **Worst Rated**.
5. Removed titles with middle-range ratings (4.0–8.0).

---

## **Methodology**

### **1. Models Implemented**
1. **Logistic Regression**:
   - Baseline model.
   - Accuracy: **78.59%**.
2. **Random Forest**:
   - Leveraged ensemble learning with 100 decision trees.
   - Accuracy: **80.55%**.
3. **Gradient Boosting**:
   - Iteratively improved weak learners.
   - Accuracy: **81.47%**.
4. **MLP Classifier**:
   - Feedforward neural network with two hidden layers.
   - Accuracy: **80.5%**.

### **2. Exploratory Data Analysis**
- **Genre Distribution**: Top 10 genres visualized using bar charts.
- **Correlation Analysis**: Moderate correlations identified using heatmaps.
- **Class Imbalance**: Balanced representation of "Top Rated" vs. "Worst Rated".

---

## **Results**

### **Model Performance**
| Model              | Accuracy (Before Tuning) | Accuracy (After Tuning) | Improvement |
|--------------------|--------------------------|--------------------------|-------------|
| Logistic Regression| 78.59%                  | 78.69%                  | +0.1%       |
| Random Forest      | 80.55%                  | 83.17%                  | +2.62%      |
| Gradient Boosting  | 81.47%                  | 83.09%                  | +1.62%      |
| MLP Classifier     | 80.50%                  | 80.44%                  | -0.06%      |

### **Insights**
- **Gradient Boosting**: Best performer with **81.47% accuracy** (83.09% after tuning).
- **Random Forest**: Significant improvement with hyperparameter tuning (**83.17% accuracy**).
- **MLP Classifier**: Neural networks struggled with this structured dataset.

---

## **Future Work**
1. **Feature Expansion**:
   - Add features like user demographics, production budget, or sentiment analysis from reviews.
2. **Deep Learning**:
   - Explore CNNs or transformers for unstructured text data.
3. **Class Imbalance Handling**:
   - Use SMOTE or weighted loss functions for imbalanced datasets.
4. **Explainability**:
   - Integrate SHAP for feature importance and model interpretability.
5. **Real-World Deployment**:
   - Create a real-time IMDb rating classification tool using APIs for metadata fetching.
6. **Ensemble Learning**:
   - Combine models using stacking or voting to improve robustness.

---

## **Usage**

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/imdb-rating-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running Models**
1. **Logistic Regression**:
   ```bash
   python logistic_regression.py
   ```
2. **Random Forest**:
   ```bash
   python random_forest.py
   ```
3. **Gradient Boosting**:
   ```bash
   python gradient_boosting.py
   ```
4. **MLP Classifier**:
   ```bash
   python mlp_classifier.py
   ```

---

## **Technologies Used**
- Python
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## **Acknowledgments**
- IMDb Dataset creators
- Scikit-learn for machine learning model implementations

