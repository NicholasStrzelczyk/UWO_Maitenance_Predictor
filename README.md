# UWO_Maitenance_Predictor

train.py TODO:

- implement logging
- (IF THERE IS TIME) try using RGB images
- (IF THERE IS TIME) try classifying 3 classes
- (IF THERE IS TIME) data augment for more robust model.

custom_ds.py TODO:

- (IF THERE IS TIME) make a self.augment() function using cv2 transforms
- (IF THERE IS TIME) make another class for RGB dataset for RGB model

custom_model.py TODO:

- (IF THERE IS TIME) create a new U-Net architecture that uses RGB images and produces output with 3 classes: dust, metal, and don't care.
- (IF THERE IS TIME) create a new architecture with MobileNet "Depthwise-Seperable Convolutions"

SYNTH_DATA_MAKER TODO:

- (IMPORTANT) make scripts to generate 4 scenario datasets:
- sc1: 1 dust spot, no maintenance (DONE)
- sc2: multiple dust spots, no maintenance (DONE)
- sc3: 1 dust spot, with periodic maintenance (DONE)
- sc4: multiple dust spots, with periodic maintenance (DONE)
-  (IF THERE IS TIME) for sc3 & sc4, make random dust cloud(s) after every time cleaning occurs.
