# UWO_Maitenance_Predictor

### CNN Model TODO:
- put synth validation dataset on server and make new list.txt for it.
- implement checkpointing

### NEW METRIC IDEA
- Try plotting fouling percentage or percentage of black pixels in target against f1 score or jaccard index for each sample.
- Why? Because this will give an idea of what kinds of targets are too difficult for the model to replicate.
- Therefore, is a good judge of what scenarios are too difficult for the model to detect.

### Realization 
- Rain data is difficult to make synthetic data from. Consider removing it.
- Increase the minimum fouling spot size (via vignette size) so that small dots (nearly impossible to detect) are no longer causing inaccuracies on test set.