# Problem Statement:
There is currently no tool that can automatically evaluate the presence of age-related facial characteristics on a person's face without having to go to a dermatologist for an examination or wait until after an experimental study has been completed. Researchers have created a system, based upon deep learning, to provide users with real-time analysis of their facial images and receive immediate recommendations based upon these results.

## Objectives: 
-Detecting and locating facial characteristics which indicate the presence of age.

-Categorise the characteristics into 4 specific categories (i.e. Wrinkled, Dark Spot, Puffy Eyes and Clear Skin) using a Convolutional Neural Network (CNN).

-Create a user-friendly web based front end by allowing users to upload images with the expected annotated outcomes and percentage predictions.

-Connect the pipeline with a backend to process uploaded images, make inferences and return annotated results.

## Technologies & Libraries Used:
-Python (v3.10.1): The primary language used for all data processing and model building.

-Pandas: Used for creating DataFrames, managing the dataset, and handling CSV files (import pandas as pd).

-Matplotlib & Seaborn: Used for generating the "Class Distribution Plot" and visualizing data balance (sns.countplot).

-Scikit-learn: Used for splitting the data into training and testing sets (train_test_split).

## Development Tools
-Visual Studio Code (VS Code): The Integrated Development Environment (IDE) used to write and run your code.

-Jupyter Notebook: Used for interactive coding and ensuring the code runs step by step.
