# Smile Detection
---
## Reference
https://github.com/meng1994412/Smile_Detection

## Project Objectives
Implement a convolutional neural network capable of detecting a person smiling or not:
* Constructed LeNet architecture from scratch.
* Trained a model on a dataset of images that contain faces of people who are smiling or not smiling.
* Developed a script to detect smile in real-time.

## Language / Packages Used
* Python 3.7
* [OpenCV](https://opencv.org/opencv-4-5-5/) 4.5.5
* [keras](https://keras.io/) 2.7.0
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
The dataset, named SMILES, comes from Daniel Hromada (check [reference](https://github.com/hromi/SMILEsmileD)) (~8000 images) and WIKI (check [reference](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki)) (~5000 images). There are totol 13,000 images in the dataset, where each image has a dimension of 64x64. And the images in the dataset are tightly cropped around the face. Data from GitHub open-source had been labed and resized but from WIKI has not. The data from WIKI has been resized, labeled again and mixtured with data from GitHub open-source.

[//]: # (Image References)

[image1]: ./dataset/SMILEs/positives/positives/2.jpg
[image2]: ./dataset/SMILEs/positives/positives/4.jpg
[image3]: ./dataset/SMILEs/positives/positives/6.jpg
[image4]: ./dataset/SMILEs/positives/positives/8.jpg
[image5]: ./dataset/SMILEs/positives/positives/10.jpg
[image6]: ./dataset/SMILEs/negatives/negatives/1.jpg
[image7]: ./dataset/SMILEs/negatives/negatives/3.jpg
[image8]: ./dataset/SMILEs/negatives/negatives/5.jpg
[image9]: ./dataset/SMILEs/negatives/negatives/7.jpg
[image10]: ./dataset/SMILEs/negatives/negatives/9.jpg
[train-plot]: ./output/train_plot.PNG
[evaluation]: ./output/evaluation.PNG

The Figure 1 shows some examples of smiling image, and Figure 2 shows some example of not smiling image.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Figure 1: Positive example of the dataset (smiling).

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Figure 2: Negative example of the dataset (not smiling).

## Results
### Build the LeNet architecture from scratch
The LeNet architecture can be found in `lenet.py` inside `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, the depth), and number of classes. In this project, the input would be (width = 32, height = 32, depth = 1, classes = 2).

Table 1 demonstrates the architecture of LeNet. The activation layer is not shown in the table, which should be one after each `CONV` layer. The `ReLU` activation function is used in the project.

| Layer Type  | Output Size  | Filter Size / Stride |
| ----------- | :----------: | -------------------: |
| Input Image | 32 x 32 x 1  |                      |
| CONV        | 28 x 28 x 4  |        5 x 5, K = 4  |
| CONV        | 24 x 24 x 8  |        5 x 5, K = 8  |
| POOL        | 12 x 12 x 16 |               2 x 2  |
| CONV        |  8 x 8 x 16  |        5 x 5, K = 16 |
| POOL        |  4 x 4 x 16  |               2 x 2  |
| FC          |     120      |                      |
| Dropout     |  0.2 (20%)   |                      |
| softmax     |      2       |                      |

Table 1: Summary of the LeNet architecture.

### Train the Smile CNN
The `train_model.py` is used for the training process. The weighted model will be saved after training ([chere here](https://github.com/meng1994412/Smile_Detection/blob/master/output/lenet.hdf5)).The saved model can be used for detecting smile in real-time later.

Figure 3 shows the plot of loss and accuracy for the training and validation set. As we can see from the figure, validation loss past 6th epoch starts to stagnate. Further training past 20th epoch may result in overfitting. Implement data augmentation on training set would be a good future "next-step" plan.

![alt text][train-plot]

Figure 3: Plot of loss and accuracy for the training and validation set.

Figure 4 illustrates the evaluation of the network, which obtains about 90% classification accuracy on validation set.

![alt text][evaluation]

Figure 4: Evaluation of the network.

### Run the Smile CNN in real-time
