# UWO_Maitenance_Predictor

### CNN Model TODO:
- train models.

### NEW METRIC IDEA
- Try plotting fouling percentage or percentage of black pixels in target against f1 score or jaccard index for each sample.
- Why? Because this will give an idea of what kinds of targets are too difficult for the model to replicate.
- Therefore, is a good judge of how well the model performs on extremely difficult samples.

### Rain Data Problem 
- Rain data is difficult to make synthetic data from.
- Consider a new data synthesis method for making rainy data.

### TODO FOR RESULTS:
- Save CSV for Binary Precision Recall Curve for each model, then plot them into one figure.
- Save CSV for Histograms 

### FOR PAPER RESULTS:
- 2 (maybe 3) experiments:
- tested 3 optimizers (with 1 figure of all precision recall curves), selected best one. (also histograms!)
- (maybe) determine how growing rate affects results. Train and test models on new datasets.
- tested best model with the original 4-scenario-dataset for more realistic PdM.