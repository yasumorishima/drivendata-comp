# tf-cpu-v1 (2026-10-23T08:07Z, commit 55da65d, TF on CPU)

memo: tf-cpu-v1（再々）: scipy 無しで落ちたのを直した。長い文で切り詰めが落ちたのを直した。Kaggle TPU が 1 時間以上空き待ちのため、同じ cross-encoder 微調整を Actions CPU で fold ごとに（長さ 160・1 epoch）

```
no xla: ModuleNotFoundError("No module named 'torch_xla'")
device cpu model cross-encoder/ms-marco-MiniLM-L-6-v2 max_len 160 epochs 1 folds [0] train rows 59882 test rows 20662 cpus 4
fold 0 step 500/1496 loss 0.4796 1570s
fold 0 step 1000/1496 loss 0.4290 3135s
fold 0: within-paper spearman tf 0.2187 vs v4 0.4277  (5599s)
done
no xla: ModuleNotFoundError("No module named 'torch_xla'")
device cpu model cross-encoder/ms-marco-MiniLM-L-6-v2 max_len 160 epochs 1 folds [1] train rows 59882 test rows 20662 cpus 4
fold 1 step 500/1497 loss 0.4155 1301s
fold 1 step 1000/1497 loss 0.4377 2584s
fold 1: within-paper spearman tf 0.2284 vs v4 0.3950  (4608s)
done
no xla: ModuleNotFoundError("No module named 'torch_xla'")
device cpu model cross-encoder/ms-marco-MiniLM-L-6-v2 max_len 160 epochs 1 folds [2] train rows 59882 test rows 20662 cpus 4
fold 2 step 500/1496 loss 0.4713 1029s
fold 2 step 1000/1496 loss 0.3973 2036s
fold 2: within-paper spearman tf 0.2224 vs v4 0.4146  (3428s)
done
no xla: ModuleNotFoundError("No module named 'torch_xla'")
device cpu model cross-encoder/ms-marco-MiniLM-L-6-v2 max_len 160 epochs 1 folds [3] train rows 59882 test rows 20662 cpus 4
fold 3 step 500/1496 loss 0.4507 1564s
fold 3 step 1000/1496 loss 0.4502 3120s
fold 3: within-paper spearman tf 0.2298 vs v4 0.4177  (5512s)
done
no xla: ModuleNotFoundError("No module named 'torch_xla'")
device cpu model cross-encoder/ms-marco-MiniLM-L-6-v2 max_len 160 epochs 1 folds [4] train rows 59882 test rows 20662 cpus 4
fold 4 step 500/1498 loss 0.4717 1318s
fold 4 step 1000/1498 loss 0.4385 2599s
fold 4: within-paper spearman tf 0.2373 vs v4 0.4393  (4559s)
done
folds merged 5, train rows 59882, test rows 20662
```
