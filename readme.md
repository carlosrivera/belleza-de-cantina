This is a pretty bad documented and totally unmaintained face alignment project.

### Usage
Build (compiling Dlib and OpenCV takes ages, be patient) and run a bash into the container:
```
docker build -t face .
docker run -ti -v $(pwd)/:/home face /bin/bash
```

Calling the API:
```
#from a path of jpg's creates a an average target.jpg
face_average(path, target)

#from a path of jpg's creates an image sequence blending pairs
face_morph(path, target)
```

### Credits
Based on [Average Face : OpenCV (C++ / Python) Tutorial](https://medium.com/@LearnOpenCV/average-face-opencv-c-python-tutorial-3a89b5347bdd)
