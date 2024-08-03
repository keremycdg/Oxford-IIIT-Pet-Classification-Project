# Oxford-IIIT Pet Classification Project

## Project Overview

This project aims to classify images from the Oxford-IIIT Pet Dataset using pre-trained models, MobileNetV2 and NASNetMobile. The goal is to demonstrate the effectiveness of transfer learning in image classification tasks.

## Dataset

The Oxford-IIIT Pet Dataset consists of images of 37 breeds of cats and dogs, with approximately 200 images per class. The dataset is balanced and commonly used for benchmarking image classification models.

## Installation

1. Clone the repository:

    git clone https://github.com/keremycdg/Oxford-IIIT-Pet-Classification-Project.git
    cd Oxford-IIIT-Pet-Classification-Project


2. Install the required packages:

    pip install -r requirements.txt


## Project Structure

- `preprocess.py`: Script for preprocessing the dataset.
- `train.py`: Script for training the models.
- `evaluate.py`: Script for evaluating the models and visualizing results.
- `fine_tuning.py`: Script for fine-tuning the models.
- `requirements.txt`: List of dependencies.
- `README.md`: Project overview and instructions.

## Preprocessing

Run the preprocessing script to resize and normalize the images:

python preprocess.py

![project](https://github.com/user-attachments/assets/64487b6b-3054-42a1-824e-61c1b4e3ef11)

