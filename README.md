Generate learning curves for drug response models.

## Keras-tuner
On the Lambda GPU cluster, the project dir is at `/lambda_stor/data/apartin/projects/dr-learning-curves`.<br>
You need the following:
- src/
- tuner.bash
- k-tuner/my-tuner.py
- data/ml.dfs/data.gdsc.dd.ge.raw
- tuner_requirements.txt

```
$ bash tuner.bash gdsc 10000 nn0
```
This will dump the results to `k-tuner/gdsc_nn0_tuner_out/tr_sz_10000`.<br>
The the best set of HPs (with some other metadata) is saved to `k-tuner/gdsc_nn0_tuner_out/tr_sz_13377/my_logs/best_hps.txt`.
