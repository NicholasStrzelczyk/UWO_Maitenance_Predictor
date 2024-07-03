# UWO_Maitenance_Predictor

train.py TODO:

- implement logging
- try using RGB images
- try classifying 3 classes

custom_ds.py TODO:

- make a self.augment() function using cv2 transforms
- make another class for RGB dataset for RGB model

custom_model.py TODO:

- create a new U-Net architecture that converts its output to B/W
- create a new U-Net architecture that uses RGB images and produces output with 3 classes: dust, metal, and don't care.
- create a new architecture with MobileNet "Depthwise-Seperable Convolutions"

# GENERAL TODOs:

- (MAIN GOAL) make synthetic dataset that shows dust growth.
- data augment for more robust model.