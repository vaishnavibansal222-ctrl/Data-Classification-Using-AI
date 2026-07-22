import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sns.set_style("whitegrid")

# ===========================
# PROJECT TITLE
# ===========================

print("=" * 65)
print("      IRIS FLOWER SPECIES CLASSIFICATION USING AI")
print("              Developed By : Vaishnavi Bansal")
print("=" * 65)

# ===========================
# CREATE OUTPUT FOLDER
# ===========================

OUTPUT_DIR = "outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# ===========================
# LOAD DATASET
# ===========================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# ===========================
# EXPORT DATASET TO CSV
# ===========================

df.to_csv(
    os.path.join(OUTPUT_DIR, "iris_dataset.csv"),
    index=False
)


print("\nDataset exported successfully.")


# ===========================
# DATASET PREVIEW
# ===========================

print("\nDataset Preview\n")
print(df.head())

# ===========================
# DATASET INFORMATION
# ===========================

print("\nDataset Information\n")
print(df.info())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ===========================
# CLASS DISTRIBUTION GRAPH
# ===========================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Species",
    hue="Species",
    palette="Set2",
    legend=False
)



plt.title("Species Distribution")
plt.xlabel("Flower Species")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "species_count.png"
    ),
    dpi=300
)

plt.show()

# ===========================
# PAIR PLOT
# ===========================

pairplot = sns.pairplot(
    df,
    hue="Species",
    diag_kind="kde",
    palette="Set2"
)

pairplot.fig.suptitle(
    "Pairwise Feature Relationships",
    y=1.02
)

pairplot.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_pairplot.png"
    ),
    dpi=300
)

plt.show()

# ===========================
# CORRELATION HEATMAP
# ===========================

plt.figure(figsize=(7,6))

numeric_df = df.drop("Species", axis=1)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.show()

# ===========================
# FEATURES AND TARGET
# ===========================

X = iris.data
y = iris.target


# ===========================
# TRAIN TEST SPLIT
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ===========================
# MODEL TRAINING
# ===========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")


# ===========================
# MODEL PREDICTION
# ===========================

y_pred = model.predict(X_test)

#===========================
#SAVE CONFUSION MATRIX
#===========================

# ===========================
# CONFUSION MATRIX
# ===========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print("----------------")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()



# ===========================
# ACCURACY
# ===========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")
print("-------------------")
print(f"{accuracy*100:.2f}%")



# ===========================
# CLASSIFICATION REPORT
# ===========================

print("\nClassification Report")
print("--------------------------")

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

#===========================
#ACCURACY VS NUMBER OF TREES
#===========================
tree_range = range(10, 201, 10)

accuracy_list = []

for trees in tree_range:

    rf = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    score = cross_val_score(
        rf,
        X,
        y,
        cv=5
    ).mean()

    accuracy_list.append(score)

plt.figure(figsize=(8,5))

plt.plot(
    tree_range,
    accuracy_list,
    marker="o"
)

plt.title("Accuracy vs Number of Trees")

plt.xlabel("Number of Trees")

plt.ylabel("Cross Validation Accuracy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_vs_trees.png"
    ),
    dpi=300
)

plt.show()

# ===========================
# FEATURE IMPORTANCE
# ===========================

importance = model.feature_importances_

feature_df = pd.DataFrame({

    "Feature": iris.feature_names,
    "Importance": importance

})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print("-----------------------")
print(feature_df)



plt.figure(figsize=(8,5))

plt.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.show()

#===========================
#CROSS VALIDATION
#===========================
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")
print(scores)

print("\nAverage Accuracy")
print(scores.mean())

#===========================
#SAVE TRAINED MODEL
#===========================
joblib.dump(
    model,
    os.path.join(
        OUTPUT_DIR,
        "iris_random_forest_model.joblib"
    )
)

print("\nModel saved successfully.")

# ===========================
# SAMPLE PREDICTION
# ===========================

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

flower = iris.target_names[prediction[0]]

print("\nSample Prediction")
print("---------------------")

print("Input Values")

print("Sepal Length :", sample[0][0])

print("Sepal Width  :", sample[0][1])

print("Petal Length :", sample[0][2])

print("Petal Width  :", sample[0][3])

print("\nPredicted Flower :", flower)

# ===========================
# PROJECT COMPLETED
# ===========================

print("\n" + "="*65)
print("Project Completed Successfully")
print("Iris Flower Classification Using Artificial Intelligence")
print("Developed By : Vaishnavi Bansal")
print("="*65)






