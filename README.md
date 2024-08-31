# UWO_Maitenance_Predictor

### TODO:
- ReTest SMS, it currently isn't working correctly!
- Plot Each Scenario seperately for SMS Test
- (Optional) create third SMS scenario for testing

### Notes from zoom session on Aug 28th:
- Look into ROC curves
- Try setting an initial bias in the model.
- Try weighted BCE loss: https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265
- Maybe restructure how train/val/test splits work: maybe have 1 dataset with 4 train scenarios, 1 val scenario, 1 test scenario.
- ^ this guarentees that a full scenario is used for validation and testing, not just any random samples.