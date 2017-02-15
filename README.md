# Gaze-Glasses-Code

If you have Docker installed:

1) Start Docker: 
```
docker run -it -v [path of where you cloned repo]/Gaze-Glasses-Code:/Gaze-Glasses-Code  gcr.io/tensorflow/tensorflow:latest-devel
```
2) In Docker: 
```
python /Gaze-Glasses-Code/label_image.py /Gaze-Glasses-Code/trainertestimage.jpg [or any other path of image]
```

If you only have TensorFlow installed:

Go to the folder and run:
```
python label_image.py trainertestimage.jpg [or any other path of image]
```

The output should tell you a confidence percentage of what it thinks the object is.


source: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4
