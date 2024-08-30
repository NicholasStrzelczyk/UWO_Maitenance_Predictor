# UWO_Maitenance_Predictor

### TODO:
- Test AdamW, Adam, SGD using new CGS test set. ---> create new graphs
- Test CGS30, CGS50, CGS using new test set. ---> create new graphs
- Test SMS.

### Notes from zoom session on Aug 28th:
- Look into ROC curves
- Try setting an initial bias in the model.
- Try weighted BCE loss: https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265
- Maybe restructure how train/val/test splits work: maybe have 1 dataset with 4 train scenarios, 1 val scenario, 1 test scenario.
- ^ this guarentees that a full scenario is used for validation and testing, not just any random samples.