# CLP-Recognition

Python 3.8, Tensorflow 2.0, OpenCv 3.4

#Notice# : notebook doesn't support cv2.imshow method, we comment all this method, if you want to use it, you can paste the program in a py file and uncomment the 

cv2.imshow methods

Please download the dataset:
https://drive.google.com/file/d/1DQJch_CUCPvSYaR-tiwuB9wVgmwuA-di/view?usp=sharing

Dataset:

Extract it and put it with three notebook files in the same folder.

It includes train and test dataset for character and license plate recognition

Folder name: char_test, char_train, plate_test, plate_train

It also includes output model file

Folder name: models (include two models for two recognition)

It also includes test images for play in main program, in a folder named test_images

We include two images for you to test, you can download other images to test as you wish. (Remember to change the path name with image name which is in main segment

of the main program.)

One more folder named opencv includes processed license plate binary image after process steps.

Program:

There are three notebooks files.

Character Model Generator and License Plate Model Generator are both model generate program.

Both program have train and test process, change the flag as 1 to train and 0 or other numbers to test.

Run both programs based on order in notebook.

CLP Recognition Main Program is the Main program for this project.

Run it based on order in notebook as well.

The trained model already include in the models folder. If you want to train it by yourself, please delete the ckpt model files first.

Also, sometimes the model name will change by the training steps, please change the model name in all the three notebook files after your own training step.


########Extra#########

We also includes a sample program to show how we split the character from license plate in splitlicenseplate folder.

Just download everything and run the program, and it will work. Outputs will in the test folder.




####Reference List:


[1] wikipedia. ”Automatic number-plate recognition”. https://en.wikipedia.org/wiki/Automatic numberplate
recognition

[2] Rosebrock, A. ”OpenCV: Automatic License.” Number Plate Recognition (ANPR) with Python: https://www.
pyimagesearch. com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python.

[3] detectRecog. ”CCPD2019”. https://github.com/detectRecog/CCPD

[4] wikipedia. ”Convolutional neural network”. https://en.wikipedia.org/wiki/Convolutional neural network

[5] Bradski, G. (2000). The OpenCV Library. Dr. Dobb’s Journal of Software Tools

[6] Mart‘n Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado,
Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey
Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg,
Dan Man‘, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit
Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vi‘gas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow:
Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
15

