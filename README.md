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

- **1. Load and Visualize Data.ipynb**: This notebook demonstrates how to load and visualize facial keypoint data from the dataset. It utilizes PyTorch's Dataset class to manage data and applies various transformations, including normalization, rescaling, and cropping. The data is then prepared for training a convolutional neural network (CNN) for facial keypoint detection.

- **2. Define the Network Architecture.ipynb**: This notebook focuses on developing a Convolutional Neural Network (CNN) using PyTorch to predict facial keypoints from image data. It defines the CNN architecture with convolutional layers, max-pooling layers, and fully-connected layers, including dropout layers to reduce overfitting. The notebook covers preprocessing of image data (rescaling, cropping, normalizing, and converting to tensors) and details the training and validation process of the CNN on a labeled dataset. Performance is visualized by comparing predicted keypoints with ground truth on test data, allowing for iterative improvements to the network.

- **3. Facial Keypoint Detection, Complete Pipeline.ipynb**: This notebook provides a complete pipeline for detecting facial keypoints using a trained CNN. It involves using a Haar Cascade face detector to locate faces in images, preprocessing the detected faces by converting them to grayscale, normalizing, rescaling, and transforming them into tensors suitable for the CNN. The trained model is then used to predict facial keypoints, with results visually compared against the original images for validation and model refinement.

## Usage (Under Construction)

### Training

To train the network:

1. Define your model architecture in a file named `models#.py`.
2. Open `Define the Network Architecture.ipynb` on Google Colab and import `Net()` from `models#.py` (your defined model) into the training section. Follow the instructions in the notebook to train and save the model.

### Testing

Use `Facial Keypoint Detection, Complete Pipeline.ipynb` to evaluate the performance of the trained model.

## Model Architecture (Under Construction)

The neural network architecture is defined in the `models.py` file, which includes convolutional layers, max-pooling layers, and fully-connected layers. Modify the architecture in `models.py` to experiment with different network structures.

## Data Transformation (Under Construction)

Data transformation, including resizing, cropping, normalization, and converting images to tensors, is handled by `data_transform` defined in `data_load.py` and detailed in the notebook **1. Load and Visualize Data.ipynb**. This ensures the data is appropriately prepared for training and testing.

## Training (Under Construction)

The training process is managed in the provided Jupyter Notebook or Python script. You can adjust parameters such as the number of epochs, batch size, loss function, and optimizer to train the model. Default settings are provided as a starting point.

## Testing (Under Construction)

Post-training, evaluate the model on test data. The test data is loaded, transformed, and processed similarly to the training data. The model's performance is assessed by comparing predicted keypoints with ground truth keypoints on test images.

## Feature Visualization (Under Construction)

The project will include a section for feature visualization, where you can explore the convolutional kernels of the trained model and observe the types of features they detect in images.

## Troubleshooting

If you encounter issues with the code, please open an issue or refer to the Wiki pages for troubleshooting tips.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is part of the Udacity Computer Vision Nanodegree program.
- Haar Cascade classifier and pre-trained models are provided by OpenCV.
- Facial keypoint detection is a challenging problem with applications in computer vision and robotics.

Feel free to explore, experiment, and enhance this project for your specific needs. Enjoy working with facial keypoint detection!

## Under Construction

- Analyze the results of the application and discuss the pros and cons of your model.
- Provide suggestions for improving the model.

