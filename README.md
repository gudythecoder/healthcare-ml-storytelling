# healthcare-ml-storytelling
Storytelling with Machine Learning: Classification, Clustering, and Regression on a Healthcare Dataset
# 🧠 Machine Learning in Healthcare: A Story-Driven Analysis

**From Literature to Logistic Regression — Combining Words and Data to Save Lives**

This project is a showcase of how I’ve blended my background in **English Literature** with my technical skills in **machine learning and analytics** to uncover patterns in patient health data. I approached it not just to build models, but to tell a story — a story about people, risk, and the power of prediction.

---

## Project Overview

I worked with a healthcare dataset that includes:

- Age, Gender, BMI
- Blood Pressure, Cholesterol Level, Heart Rate
- Blood Sugar Level, Smoking Status
- Disease Diagnosis (target)

My goal was to use **supervised**, **unsupervised**, and **hybrid machine learning** techniques to explore the data, solve real-world healthcare problems, and enhance diagnostic insights.

---

## Machine Learning Approaches

### 1.  Supervised Learning – Predicting Disease Diagnosis

**Objective:** Predict whether a patient is likely to be diagnosed with a disease based on health attributes.

**Models Used:**
- Logistic Regression
- Random Forest Classifier

**Key Metrics:**

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression|   0.513  |   0.49    |  0.26  |   0.34   |  0.505  |
| Random Forest      |   0.503  |   0.49    |  0.53  |   0.51   |  0.504  |

While both models performed close to random, the Random Forest model performed slightly better at identifying true positives. It reminded me that **even low performance carries a story** — sometimes it's a sign that features don’t hold strong predictive power, and that’s an insight worth telling.

---

### 2.  Unsupervised Learning – Identifying Patient Segments

**Objective:** Group patients into meaningful health profiles based on risk factors, without labels.

**Method:**
- K-Means Clustering
- PCA for 2D visualization
- Elbow Method for optimal cluster count

**Insights:**
- **Cluster 1**: Younger, low BMI & BP (low-risk)
- **Cluster 2**: High BP, BMI, and sugar (at-risk)
- **Cluster 3**: Older, high cholesterol (senior risk group)

This analysis gave me a new appreciation for how **data can organize itself** when given structure — and how we can find meaning in that organization.

---

### 3. Mixed Learning – Clustering + Classification

**Objective:** Improve disease diagnosis prediction by incorporating cluster membership into my supervised model.

**Steps I Took:**
1. Performed K-Means clustering to group patient profiles
2. Added the resulting `Cluster` feature to the dataset
3. Retrained Logistic Regression and Random Forest using the new feature

**Result:** Slight improvement in predictive performance — showing how **structural features from unsupervised learning** can enhance classification.

---

## Tools & Libraries

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- XGBoost (for regression)
- Jupyter Notebooks

---

## Project Structure

healthcare-ml-storytelling/ ├── data/ │ └── healthcare_dataset.csv ├── notebooks/ │ ├── Supervised.ipynb │ ├── UnsupervisedLearning.ipynb │ └── MixedLearning.ipynb ├── images/ │ ├── confusion_matrix_logistic_regression.png │ ├── confusion_matrix_random_forest_classifier.png │ ├── pca_clusters.png │ └── elbow_plot.png ├── requirements.txt └── README.md


---

## Visual Highlights

| Confusion Matrix (LogReg) | PCA Clusters |
|---------------------------|--------------|
| ![Confusion Matrix](images/confusion_matrix_logistic_regression.png) | ![PCA Plot](images/pca_clusters.png) |

---

## Reflections

This project helped me sharpen my skills in:
- Asking better data-driven questions
- Comparing multiple machine learning approaches
- Turning model performance into meaningful interpretation
- Telling stories with data

I believe in **clarity, communication, and curiosity**. And I believe that the best models are the ones that don't just predict — they explain.

> “The numbers already know the truth. I just help them tell it.”


