# ICON Implementation for WILDS2.0 Benchmark

Note: This is an anonymous repo for ICON implementation as the work is under review. I will move it to the official github repo after the paper is accepted.

## Overview
Our ICON algorithm is compatible for all 8 datasets in the paper Extending the WILDS Benchmark for Unsupervised Adaptation. The implementation is included under examples/algorithms/. We included three implementations: icon.py, icon_noaug.py and icon_det.py. icon_noaug.py is for civilcomments, amazon and ogb-molpcba, where no data augmentation is used. icon_det.py is for globalwheat. icon.py is for the rest 4 datasets. We follow the same evaluation metrics in WILDS2.0 and test our model based on the commands in the [official codalab](https://worksheets.codalab.org/worksheets/0x52cea64d1d3f4fa89de326b4e31aa50a). We combined ICON with self-training methods, i.e., noisystudent/fixmatch/pseudo-label in our WILDS2.0 experiments.

## Examples

We include the commands we used in commands.txt. We will be cleaning up the code to remove some unused or redundant parameters, so the commands will be cleaner in the future.
