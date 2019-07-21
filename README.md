<h1>Fast Face Mask Overlay<h1>
  <h3>Features:</h3>
  <ul>
    <li>Overlays a 2D mask on a detected face from a live webcam stream</li>
    <li>Detects blinks and changes masks</li>
  </ul>
 

<h3>Tools used:</h3>
<ul>
    <li>OpenCV</li>
    <li>DLIB</li>
  </ul>
  
 <h3>Instructions</h3>
 <ul>
  <li>Uses haarcascades to detect faces. Download the relevant xml from <a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml">here</a>.</li>
  <li>Uses dlib's shape predictor to detect eye landmarks and blinks. Download the predictor file from <a href="https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2">here</a>.</li>
  <li>Keep both files in the same folder as facemask.py</li>
  <li>Add more masks to the 'masks' folder if needed. Remember to have them in 1:1 aspect ratios and with white backgrounds.</li>
  <li>Consider toggling the blink time and the EAR thresholds to adjust blinkd as needed</li>
  <li>The smooth_factor parameter smoothens haarcascade's jitter. Increase it to reduce delay, at the cause of adding some unstability.</li>
  </ul>
