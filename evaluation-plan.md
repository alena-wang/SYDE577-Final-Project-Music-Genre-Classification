# End-to-end reproducibility evaluation

This run starts with the submission source and original GTZAN WAV files. Its features, split, scaler, checkpoints, and results are regenerated locally. The held-out evaluation follows the submission protocol for the shallow and deep baselines, which was fixed before training this clean copy. The four learning-rate and dropout variants contribute validation results only.

For each baseline, score the saved checkpoint from the first epoch with the highest validation accuracy for all five seeds, 577 through 581. The split seed is 577, learning rate is 0.001, and dropout is zero. Evaluate the same 150 held-out test recordings with frozen weights, no gradients, and the scaler fitted on the 699 training recordings. Keep the specifications fixed after scoring. Report test loss, accuracy, macro precision, recall, and F1 per seed and averages across seeds. Confusion matrices sum five sets of predictions on the same recordings.
