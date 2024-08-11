Abstract:
The project aims to develop a real-time method using neural networks for recognizing finger-spelling-based American Sign Language (ASL). A CNN is used to classify hand gestures captured by a camera, converting them into text and speech.

Introduction:
The focus is on building a model that recognizes finger-spelling gestures to form complete words. This is particularly useful for the Deaf and Mute (D&M) community to communicate with others.American sign language is a predominant sign language Since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and dumb(D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language.

Objectives:
Create a software that uses a CNN model to recognize ASL gestures.
Convert recognized gestures into text and speech formats.
Scope:
The system benefits both D&M individuals and those unfamiliar with sign language, enabling seamless communication.

Modules:

Data Acquisition:

Two main approaches: glove-based and vision-based.
The project opts for vision-based methods using a webcam and Mediapipe for hand detection.
The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand’s appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.

Data Preprocessing and Feature Extraction:

Hand detection using Mediapipe.
Image preprocessing using OpenCV (e.g., converting to grayscale, applying Gaussian blur, and thresholding).
The system uses landmarks from Mediapipe to draw and connect points on a plain white image.

Gesture Classification using CNN:

Gesture Classification using CNN
1. Convolutional Neural Network (CNN):
CNNs are a specialized type of neural network designed for processing structured grid data, like images. They mimic the way our brain processes visual information, particularly in the visual cortex. Here’s a detailed breakdown:

Layers in CNN:

Convolutional Layer: This is the core building block of a CNN. It applies a convolution operation to the input, passing the result to the next layer. Convolution helps in detecting features such as edges, textures, and patterns in the image by using a filter/kernel. This filter slides over the image matrix, performing dot products between the filter and the input at different spatial positions to generate a feature map.

Pooling Layer: Pooling reduces the dimensionality of each feature map but retains the most important information. It helps in making the detection of features more robust to transformations like rotation or translation of the image.

Max Pooling: Takes the maximum value from each patch of the feature map covered by the filter.
Average Pooling: Takes the average of all values from each patch of the feature map.
Fully Connected Layer (FC Layer): Unlike the convolutional layers, the fully connected layers connect every neuron in one layer to every neuron in the next layer. This layer is usually at the end of the CNN and is responsible for combining all the detected features to make a final classification.

Dimensionality in CNN:

CNN layers process data in three dimensions: width, height, and depth (number of channels, e.g., RGB for color images).
The final layer outputs a one-dimensional vector where each element corresponds to the probability of a particular class (e.g., different gesture classes).

2. Preprocessing and Classification:
   
You preprocessed 180 images for each alphabet, feeding them into the Keras CNN model.

Initially, due to poor accuracy when classifying 26 different classes, the classes were reduced to 8 groups, where each group contained similar-looking alphabets (e.g., y,j in one group).

The CNN model provides a probability distribution over these 8 classes. The label with the highest probability is treated as the predicted class.

Further classification within the group is done using mathematical operations on hand landmarks.

Accuracy: You achieved 97% accuracy with and without clean background and proper lighting, and 99% accuracy with an ideal setup.

3. Text-to-Speech (TTS) Translation:
   
The recognized gestures are translated into words, and these words are converted into speech using the pyttsx3 library in Python.
This feature simulates real-life dialogue, providing a voice output for the recognized gestures.

Project Requirements:

Hardware: A webcam for capturing gesture images.
Software: Operating System: Windows 8 or above.
IDE: PyCharm.
Programming Language: Python 3.9.
Libraries: OpenCV, NumPy, Keras, Mediapipe, TensorFlow.

System Diagrams:

1. System Flowchart:
Flow of Process:
Start by capturing the gesture using a webcam.
Preprocess the captured image (resizing, normalization, etc.).
Feed the image into the CNN model.
CNN classifies the gesture into one of the predefined classes.
If the gesture is recognized, it is translated into text.
The text is then converted into speech using TTS.
The system outputs the spoken word.
End.
2. Use-Case Diagram:
Actors: User (person making gestures), System (your application).
Use Cases:
Capture gesture.
Process gesture image.
Classify gesture.
Translate gesture to text.
Convert text to speech.
Provide feedback (voice output).
3. Data Flow Diagram (DFD):
Level 0: Capturing the gesture, processing, classification, and providing output.
Level 1: Breaking down each step in Level 0 into sub-processes, like image preprocessing, feature extraction, classification, etc.
4. Sequence Diagram:
Interaction Between Components:
User: Makes a gesture.
Webcam: Captures the gesture.
Preprocessing Module: Processes the captured image.
CNN Model: Classifies the processed image.
Text Translation Module: Converts the recognized gesture into text.
TTS Module: Converts the text into speech.
Output: Speaks the word corresponding to the gesture.
Summary:
The CNN model in your project effectively classifies hand gestures into predefined classes, achieving high accuracy. It then translates these gestures into text and further into speech using TTS, making the system user-friendly for communication.

This approach is efficient in handling visual data and is particularly beneficial in real-time gesture recognition tasks.


 
