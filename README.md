# BERT-Pruning
Tensorflow implementation of pruning on [[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)]. Currently, we fine-tune BERT pretrained model `uncased_L-12_H-768_A-12`  on several GLUE benchmark and evaluate scores on `dev` set.

The code in folder `bert`  is a clone from [google-research](https://github.com/google-research)/**[bert](https://github.com/google-research/bert)**, and we add some `DataProcessors  ` in `run_classifier.py` . The `STS-B` part code is based on [`@Colanim`](https://github.com/Colanim) 's  repo [BERT_STS-B](https://github.com/Colanim/BERT_STS-B), which use a simple regression to output scores.

### Environment

+ Python 3.7.6
+ Tensorflow 1.15.0
+ cudatoolkit 10.0.130
+ cudnn 7.6.5

### Hardware

* GTX1080ti  11GB

### Folder Description

```
BERT-Pruning
|-- bert		# from https://github.com/google-research/bert
|-- datasets	# a collection of datasets,need to download from https://gluebenchmark.com/tasks
	|-- MNLI
		|-- train.tsv
		|-- dev_matched.tsv
		|-- test_matched.tsv
	|-- MRPC
		|-- msr_paraphrase_train.txt
		|-- msr_paraphrase_test.txt
	|-- QNLI
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
	|-- QQP
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
	|-- RTE
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
	|-- SST-2
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
	|-- SST-B
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
	|-- WNLI
		|-- train.tsv
		|-- dev.tsv
		|-- test.tsv
|-- uncased_L-12_H-768_A-12	# pretained model, from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
|-- run_all.sh	# simple script to fine-tune all tasks
```

### Fine-tuning on Tasks

Here provides a simple bash script `run_all.sh` to run all tasks, each tasks will use four learning rates to fine-tune: `5e-5, 4e-5, 3e-5, 2e-5`, other hyperparameters are same as [bert](https://github.com/google-research/bert). 

```
cd /path/of/BERT-Pruning
bash run_all.sh
```

Then the folder `fine_tune_outputs` will be created in the directory. The result will be in `eval_results.txt`,  e.g.,

```
eval_accuracy = 0.91019046
eval_loss = 0.33257872
global_step = 34110
loss = 0.3325944
train_time: 304.109253min
eval_time: 4.763651min
```

### Results (dev set)

|                   | MNLI-m (Acc) | QQP (F1) | QNLI (Acc) | SST-2 (Acc) | STS-B (Spearman correlations) | MRPC (F1) | RTE (Acc) | WNLI (Acc) |
| ----------------- | ------------ | -------- | ---------- | ----------- | ----------------------------- | --------- | --------- | ---------- |
| Our results       |              |          | 91.3       | 93.2        | 77.9 (pearson: 89.6)          |           | 67.1      | 56.3       |
| **Paper results** | 84.6         | 71.2     | 90.5       | 93.5        | 85.8                          | 88.9      | 66.4      | None       |
| Training time     | 5.74h        | 5.06h    | 1.47h      | 0.98h       | 0.09h                         | 0.08h     | 0.05h     | 0.02h      |

The paper's result is **evaluated by GLUE evaluation server**（BERT base, not sure whether cased or uncased), probably test **on test set**. Since GLUE data set has no test label, we just evaluate these tasks **on dev set**, which means the result on dataset might be **lower** than current result, thus the comparison of results is for reference only.

The reason for paper **not including WNLI** result is because GLUE webpage notes that there are issues with the construction of this dataset, thus authors consider it as a problematic one.

Note that our result on **STS-2** is not good compare to paper's result on Spearman correlations metric. The most likely cause is a **different implementation of the output layer**. Since **the output of STS-B is a float type**, the paper does not seem to explain in detail how to deal with this situation, thus we just use [`@Colanim`](https://github.com/Colanim) 's idea by using a simple regression as output layer.

Here we follow paper's instructions, fine-tuning model in four different learning rates (results above are highest in each task):

| Learning Rate | MNLI-m (Acc) | QQP (F1) | QNLI (Acc) | SST-2 (Acc) | STS-B (Spearman correlations) | MRPC (F1) | RTE (Acc) | WNLI (Acc) |
| ------------- | ------------ | -------- | ---------- | ----------- | ----------------------------- | --------- | --------- | ---------- |
| 2e-5          |              |          | 91.2       | 92.4        | 77.7                          |           | 67.1      | 56.3       |
| 3e-5          | 83.9         |          | 91.3       | 93.2        | 78.0                          |           | 65.7      | 56.3       |
| 4e-5          |              |          | 91.2       | 92.2        | 78.3                          |           | 64.6      | 56.3       |
| 5e-5          |              |          | 91.2       | 91.3        | 75.6                          |           | 52.7      | 43.7       |

### Cite

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
@inproceedings{wang2019glue,
  title={ {GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
```

