# Blink Detetcion 

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Contributors](#contributors)
* [Resources](#resources)


<!-- ABOUT THE PROJECT -->
## About The Project 
|![Visualizing the 68 facial landmark coordinates](https://github.com/prakash-2702/Facial_Landmarks_Detection/blob/master/assets/HOG.PNG)|![The 6 facial landmarks associated with the eye.](https://github.com/prakash-2702/Blink_Detection/blob/master/eye.PNG)|
|:---:|:---:|
|Visualizing the 68 facial landmark coordinates|The 6 facial landmarks associated with the eye|  
* Facial landmarks are used to localize and represent salient regions of the face, such as:
  1. Eyes
  2. Eyebrows
  3. Nose
  4. Mouth
  5. Jawline
* Blink detection is actually the process of using computer vision to firstly detect a face, with eyes, and 
  then using a video stream (or even a series of rapidly-taken still photos) to determine whether those eyes 
  have blinked or not within a certain timeframe.
  
**Steps followed in this process:**
  1. Initializing dlib's face detector (HOG-based).
  2. Grabing the indexes of the facial landmarks for the left and right eye.
  3. Starting the videostream.
  4. Pre-processing of the frames(loading,resizing,converting to gray-scale).
  5. Detection of faces in the grayscale image.
  4. Determining the facial landmarks for the face region.
  5. Extracting the left and right eye coordinates.
  6. Computing the EYE ASPECT RATIO
  7. Showing the number of blinks i.e output.

### File Structure
    .
    ├── detect_blinks.py                       # Driver code
    ├── shape_predictor_68_face_landmarks.dat  # Pre-trained model link
    ├── .gitattributes
    └── README.md 
    
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites  
* Python
* OpenCV
* Numpy 
* Dlib 
* Imutils
* Argparse

### Installation
1. Clone the repo
```sh
git clone https://github.com/prakash-2702/Blink_Detection.git
```    
<!-- CONTRIBUTORS -->
## Contributors
* [Prakash Nadgeri](https://github.com/prakash-2702)
<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Resources
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
* [Histogram of Oriented Gradients and Object Detection](https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
* [pyimagesearch](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)



 
