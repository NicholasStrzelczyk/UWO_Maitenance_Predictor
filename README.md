# UWO_Maitenance_Predictor

### CNN Model TODO:
- create eval process script for testing (fouling counter, eval process, new timeline column, etc.)
- put synth dataset on server and make new list.txt for it.
- ensure deterministic seed is working!
- figure out how to weight classes correctly! maybe time for custom loss function?
- add more torchmetrics: 
- https://lightning.ai/docs/torchmetrics/stable/classification/precision_recall_curve.html
- https://lightning.ai/docs/torchmetrics/stable/classification/precision_recall_curve.html#torchmetrics.classification.BinaryPrecisionRecallCurve
- https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html
- https://lightning.ai/docs/torchmetrics/stable/segmentation/generalized_dice.html