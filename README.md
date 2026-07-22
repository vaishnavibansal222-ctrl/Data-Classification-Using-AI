🌸 Iris Flower Species Classification Using AI

A complete Machine Learning project that classifies Iris flowers into three different species using the Random Forest Classification algorithm. This project demonstrates the complete AI workflow, from data exploration to model deployment, and was developed as part of the DecodeLabs Artificial Intelligence Internship.

📖 Project Overview

Machine Learning has become an essential part of Artificial Intelligence by enabling computers to make intelligent decisions from data. This project applies Machine Learning techniques to classify Iris flowers into three different species using their physical characteristics.

The project follows a complete end-to-end Machine Learning pipeline, including:

*Loading and exploring the dataset
*Data preprocessing
*Data visualization
*Model training using Random Forest
*Performance evaluation
*Cross-validation
*Feature importance analysis
*Saving the trained model
*Predicting unseen flower species

The goal is to demonstrate a production-style machine learning workflow while maintaining clean, readable, and professional Python code.

🌺 Iris Flower Species

The model predicts one of the following flower species:
*🌸 Iris Setosa
*🌼 Iris Versicolor
*🌷 Iris Virginica
📊 Dataset Information

The project uses the famous Iris Dataset, one of the most popular datasets for Machine Learning.

*Dataset Statistics
*Total Samples: 150
*Features: 4
*Target Classes: 3

Features Used
| Feature      | Description              |
| ------------ | ------------------------ |
| Sepal Length | Length of the sepal (cm) |
| Sepal Width  | Width of the sepal (cm)  |
| Petal Length | Length of the petal (cm) |
| Petal Width  | Width of the petal (cm)  |

🚀 Technologies Used
*Python
*NumPy
*Pandas
*Matplotlib
*Seaborn
*Scikit-learn
*Joblib

🤖 Machine Learning Algorithm
Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to achieve high accuracy while reducing overfitting.

Why Random Forest?

✔ High Prediction Accuracy

✔ Handles Non-linear Data

✔ Less Overfitting

✔ Provides Feature Importance

✔ Fast and Reliable

📂 Project Structure
Iris-Flower-Classification-Using-Random-Forest/
│
├── outputs/
│   ├── iris_dataset.csv
│   ├── species_count.png
│   ├── feature_pairplot.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── accuracy_vs_trees.png
│   ├── feature_importance.png
│   └── iris_random_forest_model.joblib
│
├── screenshots/
│   ├── species_count.png
│   ├── pairplot.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── iris_classifier.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── assets/
    └── banner.png

🔄 Project Workflow

                Iris Dataset
                      │
                      ▼
             Data Exploration
                      │
                      ▼
             Data Visualization
                      │
                      ▼
             Train-Test Split
                      │
                      ▼
        Random Forest Training
                      │
                      ▼
          Model Prediction
                      │
                      ▼
         Performance Evaluation
                      │
                      ▼
         Cross Validation
                      │
                      ▼
       Feature Importance Analysis
                      │
                      ▼
          Save Trained Model
                      │
                      ▼
            Predict New Samples

    📈 Generated Visualizations

The project automatically generates the following visualizations:

*📊 Species Distribution
*📉 Pair Plot
*🔥 Correlation Heatmap
*📋 Confusion Matrix
*📈 Accuracy vs Number of Trees
*📊 Feature Importance

🖼 Project Screenshots

## Species Distribution

![Species Distribution](screenshots/species_count.png)


## Pair Plot

![Pair Plot](screenshots/pairplot.png)

## Correlation Heatmap

![Heatmap](screenshots/heatmap.png)


## Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

## Feature Importance

![Feature Importance](screenshots/feature_importance.png)

📊 Model Evaluation

The trained model is evaluated using:

*Accuracy Score
*Classification Report
*Confusion Matrix
*Cross Validation
*Feature Importance
*Sample Prediction

📁 Output Files

The project automatically generates:

*iris_dataset.csv
*species_count.png
*feature_pairplot.png
*correlation_heatmap.png
*confusion_matrix.png
*accuracy_vs_trees.png
*feature_importance.png
*iris_random_forest_model.joblib

💻 Installation
git clone https://github.com/vaishnavibansal222-ctrl/Data-Classification-Using-AI.git

cd Data-Classification-Using-AI

pip install -r requirements.txt

python "PROJECT-2 VAISHNAVI DATA CLASSIFICATION USING AI.py"

📌 Future Improvements
*Streamlit Web Application
*Flask REST API
*Hyperparameter Optimization
*Compare Multiple Machine Learning Models
*Deploy on Hugging Face Spaces
*Interactive Dashboard
*Model Explainability using SHAP

🎯 Skills Demonstrated
*Machine Learning
*Artificial Intelligence
*Data Analysis
*Data Visualization
*Feature Engineering
*Model Training
*Cross Validation
*Random Forest Classification
*Model Serialization
*Python Programming
*Git & GitHub

👩‍💻 Developed By

Vaishnavi Bansal

DecodeLabs Artificial Intelligence Internship

📜 License

This project is licensed under the MIT Lice

⭐ Show Your Support

If you found this project useful, please consider giving it a ⭐ Star on GitHub.
