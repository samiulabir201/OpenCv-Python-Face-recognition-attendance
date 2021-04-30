# OpenCv-Python-Face-recognition-attendance
#necessary dependencies to be installed:
cmake,
dlib(19.18.0),
face-recognition,
numpy,
OpenCV

In this project I tried to perform facial recognition test with high accuracy result. The project also create a real-time attendance result of the persons after facial recognition. This is my first OpenCV python project. So I think there’s a lot more room to improve. 

A short brief is given below:
I used a face recognition library by Adam Geitgey to recognize the faces and differentiate between different people. Step one is to find the faces at the backend and the method used here is known as hogg(histogram of oriented gradients). After getting hogg sample and images in a bounding box , the program uses dlib library at the back-end for the network to understand human faces. Then the image is sent to the previously trained neural network which gives us the encoded features. And lastly it used the SVM classifier machine learning method to  method  to differentiate between different people and to find the matches . 

The link of the article:

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
