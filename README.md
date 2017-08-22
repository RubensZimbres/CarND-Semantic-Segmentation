# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Write Up
* The main implementation is in the `main.py`.
* You can find the files regarding the saved model, and optimised model [here](https://drive.google.com/drive/folders/0Bxx8osZ5EmIiNFY1cTZZQVRmSDg?usp=sharing). If you want to reuse it, download them in a folder called `model` in git folder.
* I generated as well the 8bit version of the network. The file is called `eightbit_graph.pb` shared in goolge drive folder. You can run the `OnlyInference.py` to use this 8bit graph and the result of this model is saved in [here](https://github.com/yosoufe/CarND-Semantic-Segmentation/tree/master/run2). Some of the images are done poorly using the 8bit model but a lot of them are almost the same. The `.pb` file is hugely smaller.
* I have tried to optimise using the tensorflow optimiser. The output was not that much different from the input in terms of size.

### My Learnings
* I had a difficult time before get everything working. The main was the model was not getting trained. The main problem was I was choosing ig learning rate. That was causing the network stop training at first early stages.
* In order to have a network, compatible to different shapes, I used `tf.slice` and `tf.shape` operations.
