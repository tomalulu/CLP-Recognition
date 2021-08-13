# CLP-Recognition
Please download the dataset:
https://drive.google.com/file/d/1DQJch_CUCPvSYaR-tiwuB9wVgmwuA-di/view?usp=sharing

Dataset:

Extract it and put it with three notebook files in the same folder.

It includes train and test dataset for character and license plate recognition

Folder name: char_test, char_train, plate_test, plate_train

It also includes output model file

Folder name: models (include two models for two recognition)

It also includes test images for play in main program, in a folder named test_images

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

Just download everything and run the main.py program, and it will works.




Reference List:


[1] Rosebrock, A. ”OpenCV: Automatic License.” Number Plate Recognition (ANPR) with Python: https://www.
pyimagesearch. com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python.

[2] detectRecog. ”CCPD2019”. https://github.com/detectRecog/CCPD

[3] wikipedia. ”Convolutional neural network”. https://en.wikipedia.org/wiki/Convolutional neural network

[4] Bradski, G. (2000). The OpenCV Library. Dr. Dobb’s Journal of Software Tools

[5] Mart‘n Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado,
Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey
Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg,
Dan Man‘, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit
Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vi‘gas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow:
Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

