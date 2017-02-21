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

1) Change the paths in label_image.py:

From `/Gaze-Glass-Code/label_image.py` to `label_image.py`

2) Run:
```
python label_image.py trainertestimage.jpg [or any other path of image]
```

The output should tell you a confidence percentage of what it thinks the object is.


source: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4
