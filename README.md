# Semantic-Segmentation-of-Magnetic-Resonance-MR-Images-using-Deep-Learning

**Intro**
This is a group coursework done as part of my neural computation module during my master's at the university of Birmingham.

**Disclosure**:
This was group project.
My personal part was the selection of the architecture of the CNN and the implementation of it to accomplish the desired task.

**Outcome of file**:
The files have been tested and do run acoordingly.

**Task**:
CW.pdf is the file given by the univesity.
TLDR:
This task is about semantic segmentation of magnetic resonance images using deep learning. The goal of the task is segmentation of the cardiovascular MR images into 4 catogories. These being the background region(black), the left ventricle region(white), the myocardium region (white grey) and the right ventricle region (dark grey).

The goal is to create an automated methodology using Convolutional Neural Network (CNN) that is capable of segmenting the CMR image into the four regions. The choice of the architecture to use was up to us.

**Solution**:
The architecture used was the UNET architecture as it has shown promise for this type of segmentation task.

**Row to run** :
The main.py file will to the training and when it is complete will display the images and save the network in the networks folder.

**Results**:
The outcome of this is found in the data/test/mask folder. These files are generated from the main using the test images in the image folder as the input.
