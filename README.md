# Gaze-Glasses-Code

Assuming Docker is installed:
1) Start Docker: 
  $ docker run -it -v [path of where you cloned repo]/Gaze-Glasses-Code:/Gaze-Glasses-Code  gcr.io/tensorflow/tensorflow:latest-devel
2) In Docker: 
  $ python /Gaze-Glasses-Code/label_image.py [path of image you want to compare]

The output should tell you a confidence percentage of what it thinks the object is.

source: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4
