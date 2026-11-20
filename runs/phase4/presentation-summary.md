# SYDE 577 presentation summary

- **Setup.** 999 GTZAN recordings, 71 features, and one duplicate-aware split. Forty training runs covered eight configurations and five seeds. Twenty-five frozen checkpoints were evaluated on the held-out test split under a written plan.
- **Depth.** The intermediate model led validation at 70.13% ± 1.10 pp. On test, shallow and intermediate were nearly tied at 68.27% and 68.13%. The deep baseline averaged 66.13%.
- **Model size.** The parameter-matched shallow model averaged 71.07% ± 2.56 pp on test, compared with 66.13% ± 2.33 pp for deep. The models differ by 54 parameters and distribute capacity across layers differently.
- **Regularization.** Dropout 0.2 averaged 70.00% on validation and 64.53% on test. The apparent gap reduction was smaller after scoring training and validation in the same evaluation mode.
- **Takeaway.** On this fixed split, additional layers did not produce higher test accuracy. Results are descriptive. The validation and test sets each have 15 recordings per genre, and seed variability does not capture variation across data splits.

[Validation and test figure](validation-vs-test.png) · [Validation-only tuning figure](validation-tuning.png) · [Held-out confusion matrices](test-confusion.png)
