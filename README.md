# Machine-Learning-Model


# User Behavior Classification Pipeline with PySpark

This project demonstrates a scalable machine learning pipeline for classifying user behavior based on device usage data. The pipeline leverages PySpark MLlib for data preprocessing, model training, and hyperparameter tuning. 

## Project Overview

This PySpark project builds an end-to-end machine learning pipeline for multiclass classification of user behavior. The data is preprocessed, encoded, and standardized before training a RandomForestClassifier. Hyperopt is used for hyperparameter tuning to optimize model performance.

## Dataset

The dataset consists of user device usage metrics such as:
- `Device Model`
- `Operating System`
- `App Usage Time`
- `Screen On Time`
- `Battery Drain`
- `Number of Apps Installed`
- `Data Usage`
- `Age`
- `Gender`
- `User Behavior Class` (target label for classification)

The target label is renamed as `label` for processing in PySpark.

## Pipeline Steps

1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Preprocessing**:
   - **StringIndexers** for categorical features (`Device Model`, `Operating System`, `Gender`).
   - **OneHotEncoding** of the indexed features.
   - **VectorAssembler** to combine numerical and encoded features into a feature vector.
   - **StandardScaler** to normalize the feature vector.
3. **Model Training**:
   - `RandomForestClassifier` is used as the classifier.
   - The pipeline is trained on an 80/20 train-test split of the data.
4. **Hyperparameter Tuning** (Optional):
   - Hyperopt is used for distributed hyperparameter tuning on parameters like `numTrees`, `maxDepth`, and `maxBins`.
5. **Model Evaluation**:
   - A `MulticlassClassificationEvaluator` calculates model accuracy on the test set.

## Project Structure

- **`data/`**: Contains the input dataset (`user_behavior_dataset.csv`).
- **`notebooks/`**: Jupyter notebooks for data exploration and analysis.
- **`src/`**: Source code for the pipeline.
- **`README.md`**: Project documentation.
- **`requirements.txt`**: Python dependencies for running the project.

## Setup Instructions

### Prerequisites

- Python 3.7+
- PySpark
- Hyperopt
- (Optional) Jupyter Notebook for interactive testing

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start a Jupyter Notebook session if desired:

   ```bash
   jupyter notebook
   ```

### Running the Pipeline

To run the pipeline, follow these steps:

1. **Load and preprocess data**:
   The dataset (`user_behavior_dataset.csv`) is loaded and preprocessed. Ensure that the file path is correct in the code.

2. **Train the model**:
   The pipeline uses a `RandomForestClassifier` to classify user behavior based on device usage patterns.

3. **Hyperparameter tuning** (optional):
   Use Hyperopt to tune model hyperparameters and optimize performance.

4. **Evaluate the model**:
   Evaluate the trained model on the test data using `MulticlassClassificationEvaluator`.

Example:

```python
# Load the dataset
data = spark.read.csv("/path/to/user_behavior_dataset.csv", header=True, inferSchema=True)

# Train and evaluate the model
final_model = full_pipeline.fit(train_data)
predictions = final_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy}")
```

## Results

The model's performance is evaluated using accuracy as the metric. Experiment with hyperparameter tuning and different configurations to further improve accuracy.
