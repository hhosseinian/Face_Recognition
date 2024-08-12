# Facial Keypoint Detection

<a target="_blank" href="https://colab.research.google.com/github/hhosseinian/Face_Recognition">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This project aims to detect faces in images and predict facial keypoints. It involves training a neural network to detect facial keypoints, and using computer vision techniques to transform images of faces. 

### Examples of Facial Keypoints

Let’s explore some examples of images with corresponding facial keypoints.

![Facial Keypoints Example](images/key_pts_example.png)

In the image above, **facial keypoints** (or facial landmarks) are depicted as small magenta dots on each face. Each face in the dataset has **68 keypoints**, each with coordinates (x, y). These keypoints highlight significant areas of the face, including the eyes, corners of the mouth, and the nose.

These keypoints are essential for various applications, such as face filters, emotion recognition, and pose estimation. The numbered keypoints in the image below illustrate how specific ranges of points correspond to different facial features.

![Numbered Facial Landmarks](images/landmarks_numbered.jpg)


The following steps outline the process:

**Detect Faces**: Utilize a Haar Cascade detector to locate faces within images.

**Pre-process Faces**: Convert the detected faces to grayscale and transform them into a suitable input tensor format for the neural network.

**Predict Keypoints**: Apply a trained CNN to predict facial keypoints (such as eyes, nose, mouth corners) on each detected face.

## Getting Started

Before running the code, make sure you have the required libraries installed. You can install them using pip:
```
pip install numpy matplotlib opencv-python torch
```
## Project Structure
- The Jupyter notebook **1. Load and Visualize Data.ipynb** outlines the process of loading and visualizing facial keypoint data from a dataset, using PyTorch's Dataset class to manage the data, applying various transformations such as normalization, rescaling, and cropping, and preparing the data for training a convolutional neural network (CNN) for facial keypoint detection.
- The Jupyter notebook **2. Define the Network Architecture.ipynb** develops a Convolutional Neural Network (CNN) using PyTorch to predict facial keypoints from image data. The CNN architecture was defined with convolutional, max-pooling, and fully-connected layers, including dropout layers to mitigate overfitting. Image data was preprocessed with transformations such as rescaling, cropping, normalizing, and converting to tensors, ensuring consistency and efficiency during model training. The CNN was trained and validated on a labeled dataset, with hyperparameters adjusted to optimize performance. Model performance was visualized by comparing predicted keypoints with ground truth keypoints on test data, enabling iterative improvements to the network structure.
- The Jupyter notebook **3. Facial Keypoint Detection, Complete Pipeline.ipynb** outlines a comprehensive process for detecting facial keypoints using a trained Convolutional Neural Network (CNN). It involves using a Haar Cascade face detector to locate faces in images, preprocessing those detected faces by converting them to grayscale, normalizing, rescaling, and transforming them into tensors compatible with the CNN input. The notebook then applies the trained CNN model to predict facial keypoints, which are visually compared with the original image for validation and further improvement of the model.



## Usage (Under construction)

### Training
To train the network, follow these steps:

1. Define your model architecture in a separate file name as **models#.py**.
2. Open the Define the Network Architecture.ipynb on google colab and import 'Net()' the **models#.py** (your defined model) in the training part. Follow the instructions in the Jupyter notebook to train and save the model.
### Testing
  Use **Facial Keypoint Detection, Complete Pipeline.ipynb** to evaluate the performance of trained model.

## Model Architecture (Under construction)

The neural network architecture is defined in the `models.py` file. It includes convolutional layers, max-pooling layers, and fully-connected layers. You can modify the architecture in `models.py` to experiment with different network structures.

## Data Transformation (Under construction)

Data transformation, including resizing, cropping, normalization, and converting images to tensors, is applied using the `data_transform` defined in `data_load.py` and also the first notebook **1. Load and Visualize Data.ipynb**. This transformation prepares the data for training and testing.

## Training (Under construction)

The training process is handled in the provided Jupyter Notebook or Python script. You can adjust the number of epochs, batch size, loss function, and optimizer to train the model. The default settings are provided as a starting point.

## Testing (Under construction)

After training, you can evaluate the model on test data. Test data is loaded, transformed, and processed similarly to the training data. The model's performance can be visualized by comparing predicted keypoints with ground truth keypoints on test images.

## Feature Visualization (Under construction)

The project also includes a feature visualization section, where you can explore the convolutional kernels of the trained model and see what types of features they detect in an image.

## Troubleshooting
please keep me posted if you need to reaolve a problem with code. I will keep troubleshootings posted in WiKi pages.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is part of the Udacity Computer Vision Nanodegree program.
- Haar Cascade classifier and pre-trained models are provided by OpenCV.
- Facial keypoint detection is a challenging problem with applications in computer vision and robotics.

Feel free to explore, experiment, and enhance this project for your specific needs. Enjoy experimenting with facial keypoint detection!

## Under Construction
- Analyse the result of the application and discuss the pros and cons of your model.
- provide ways to improve the model.
