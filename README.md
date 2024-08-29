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

### Notes from zoom session on Aug 28th:
- Look into ROC curves
- Try setting an initial bias in the model.
- Try weighted BCE loss: https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265
- Maybe restructure how train/val/test splits work: maybe have 1 dataset with 4 train scenarios, 1 val scenario, 1 test scenario.
- ^ this guarentees that a full scenario is used for validation and testing, not just any random samples.