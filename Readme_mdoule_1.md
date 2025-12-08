Module 1: Dataset Preparation - README
What We Did
1. Setup and Exploration
Installed required Python packages (pandas, matplotlib, OpenCV, scikit-learn)

Located and examined the dataset folder structure

Found 4 skin condition categories in the dataset

2. Data Analysis
Counted images in each category:

Wrinkles: [X] images

Dark Spots: [X] images

Puffy Eyes: [X] images

Clear Skin: [X] images

Created a bar chart showing image distribution

Total images: [X]

3. Image Processing
Created function to load and resize images to 224x224 pixels

Converted images from BGR to RGB color format

Normalized pixel values to range 0-1

Tested on sample images to verify processing works

4. Data Augmentation
Set up basic image augmentation:

Random horizontal flips

Random brightness adjustments

Tested augmentation on sample images

5. Dataset Splitting
Split dataset into three parts:

Training set: 70% of images

Validation set: 15% of images

Test set: 15% of images

Organized into separate folders for each split

6. Final Outputs Created
skin_dataset_catalog.csv - Contains all image file paths and labels

partitioned_dataset/ folder with train/val/test splits

Class distribution charts showing image counts

Dataset partition visualization showing split distribution

Files Created
skin_dataset_catalog.csv - Main labels file

dataset_partition_analysis.png - Visualization of splits

partitioned_dataset/ - Organized train/val/test folders

Next Steps (Module 2)
Load the prepared dataset for training

Build CNN model for skin condition classification

Train and evaluate the model

Save the trained model for deployment