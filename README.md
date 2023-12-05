# Facial Key point Detection

This project aims to detect faces in images and predict facial keypoints. It involves training a neural network to detect facial keypoints, which can then be applied to any image containing faces. The following steps outline the process:

1. Detect all the faces in an image using a face detector (Haar Cascade detector is used in this project).
2. Pre-process the detected faces by converting them to grayscale and transforming them into a tensor of the input size required by the neural network.
3. Use the trained neural network to detect facial keypoints on the image.

## Getting Started

Before running the code, make sure you have the required libraries installed. You can install them using pip:
'''pip install numpy matplotlib opencv-python torch'''

## Usage

1. Select an image from the `images/` directory on which you want to perform facial keypoint detection.

2. Run the Jupyter Notebook or Python script provided to perform the following tasks:

   - Load necessary libraries and select an image.
   - Detect all faces in the image using a Haar Cascade classifier.
   - Load a trained facial keypoint detection model.
   - Transform each detected face into an input tensor for the model.
   - Use the model to detect and display predicted keypoints on each detected face.

## Model Architecture

The neural network architecture is defined in the `models.py` file. It includes convolutional layers, max-pooling layers, and fully-connected layers. You can modify the architecture in `models.py` to experiment with different network structures.

## Data Transformation

Data transformation, including resizing, cropping, normalization, and converting images to tensors, is applied using the `data_transform` defined in `data_load.py`. This transformation prepares the data for training and testing.

## Training

The training process is handled in the provided Jupyter Notebook or Python script. You can adjust the number of epochs, batch size, loss function, and optimizer to train the model. The default settings are provided as a starting point.

## Testing

After training, you can evaluate the model on test data. Test data is loaded, transformed, and processed similarly to the training data. The model's performance can be visualized by comparing predicted keypoints with ground truth keypoints on test images.

## Feature Visualization

The project also includes a feature visualization section, where you can explore the convolutional kernels of the trained model and see what types of features they detect in an image.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is part of the Udacity Computer Vision Nanodegree program.
- Haar Cascade classifier and pre-trained models are provided by OpenCV.
- Facial keypoint detection is a challenging problem with applications in computer vision and robotics.

Feel free to explore, experiment, and enhance this project for your specific needs. Enjoy experimenting with facial keypoint detection!
