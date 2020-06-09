# BERT-Pruning
Tensorflow implementation of pruning on [[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)]. Currently, we fine-tune BERT pretrained model `uncased_L-12_H-768_A-12`  on several GLUE benchmark and evaluate scores on `dev` set.

The code in folder `bert`  is a clone from [google-research](https://github.com/google-research)/**[bert](https://github.com/google-research/bert)**, and we add some `DataProcessors  ` in `run_classifier.py` . The `STS-B` part code is based on [`@Colanim`](https://github.com/Colanim) 's  repo [BERT_STS-B](https://github.com/Colanim/BERT_STS-B), which use a simple regression to output scores.

### Environment

+ Ubuntu 16.04 LTS
+ gcc 5.4.0

+ cudatoolkit 10.0.130
+ cudnn 7.6.5
+ Python 3.7.6
+ Tensorflow 1.15.0

### Hardware

* GTX1080ti  11GB
* TITAN RTX 24GB

### Folder Description

```
BERT-Pruning
|-- bert	# from https://github.com/google-research/bert
|-- flop    	# flop pruning method's code
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
|-- uncased_L-12_H-768_A-12_f 	# factorized model 
|-- uncased_L-12_H-768_A-12_SST-2_f
|-- run_all.sh			# simple script to fine-tune all tasks
|-- run.sh			# pruning factorized model
|-- factorize.sh		# factorize a BERT model into a new model
|-- remove_mask.sh		# remove mask layer from factorized model
```

## Fine-tuning on Tasks (Unfactorized Model)

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
| Our results       | 84.3         | 88.2     | 91.8       | 93.0        | 78.6 (pearson: 89.6)          | 89.1      | 68.2       | 60.5       |
| **Paper results** | 84.6         | 71.2     | 90.5       | 93.5        | 85.8                          | 88.9      | 66.4      | None       |
| Training time     | 5.77h        | 5.17h    | 1.48h      | 0.93h       | 0.09h                         | 0.08h     | 0.05h     | 0.03h      |

The paper's result is **evaluated by GLUE evaluation server**（BERT base, not sure whether cased or uncased), probably test **on test set**. Since GLUE data set has no test label, we just evaluate these tasks **on dev set**, which means the result on dataset might be **lower** than current result, thus the comparison of results is for reference only.

The reason for paper **not including WNLI** result is because GLUE webpage notes that there are issues with the construction of this dataset, thus authors consider it as a problematic one.

Note that our result on **STS-2** is not good compare to paper's result on Spearman correlations metric. The most likely cause is a **different implementation of the output layer**. Since **the output of STS-B is a float type**, the paper does not seem to explain in detail how to deal with this situation, thus we just use [`@Colanim`](https://github.com/Colanim) 's idea by using a simple regression as output layer.

Here we follow paper's instructions, fine-tuning model in four different learning rates (results above are highest in each task):

| Learning Rate | MNLI-m (Acc) | QQP (F1) | QNLI (Acc) | SST-2 (Acc) | STS-B (Spearman correlations) | MRPC (F1) | RTE (Acc) | WNLI (Acc) |
| ------------- | ------------ | -------- | ---------- | ----------- | ----------------------------- | --------- | --------- | ---------- |
| 2e-5          | 84.2         | 87.9     | 91.8       | 92.8        | 78.6                          | 88.1      | 68.2      | 43.7       |
| 3e-5          | 84.2         | 88.0     | 91.3       | 92.3        | 77.6                          | 88.8      | 67.1      | 43.7       |
| 4e-5          | 84.3         | 88.2     | 90.5       | 93.0        | 77.3                          | 89.1      | 52.7      | 45.1       |
| 5e-5          | 83.9         | 87.9     | 91.0       | 91.3        | 77.0                          | 85.5      | 61.7      | 60.6       |

The experimental data is stored in the folder [`fine_tune_results`](https://github.com/Holldean/BERT-Pruning/tree/master/fine_tune_results).

## Structured Pruning 

The algorithm I implement is from paper [[1910.04732]Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732), and many code is taken from their repository [asappresearch/flop](https://github.com/asappresearch/flop), however some details many be different from them. I also refer to Goggle's l0 regularization pruning code: [google-research/google-research/state_of_sparsity](https://github.com/google-research/google-research/tree/master/state_of_sparsity). All pruning code is placed under the `flop` folder.

This algorithm need to follow these four steps:

1. Factorize the matrix of each dense layer of BERT pretrain model into two submatrix.
2. Place a pruning mask diagonal matrix between every two factorized matrixes, and construct a new intermediate model.
3. Finetune this intermediate model on down steam task.
4. Remove pruning masks from factorized layer.

However, the result of this method mentioned in paper is not good, so I use another way:

1. Finetune BERT pretrain model in dataset (SST-2, learning rate 3e-5).
2. Factorize the matrix of each dense layer of this finetuned checkpoint into two submatrix.
3. Finetune this intermediate model on down steam task (set model learning rate to 0).
4. Remove pruning masks from factorized layer.
5. Finetune again (set model learning rate to 1e-6).

### 1. Factorization

In first two steps, we need download a BERT checkpoint first, fine-tune it on dataset(SST-2), run the script `factorize.sh`:

```python
python ./flop/factorize.py \
  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
  --checkpoint=./path/to/finetuned/model/bert_model.ckpt \
  --output_dir=./uncased_L-12_H-768_A-12_f/bert_model_f.ckpt \
  --finetuned
```

This script will first build a intermediate model,  then load tensors from BERT's checkpoint and factorize dense layer's matrixes, save these tensor into intermediate model. If run correctly, the following message will be shown in terminal:

```
INFO:tensorflow:Tensor: bert/encoder/layer_3/attention/self/value_p/kernel:0 *INIT_FROM_CKPT*
INFO:tensorflow:Tensor: bert/encoder/layer_3/attention/self/value_q/kernel:0 *INIT_FROM_CKPT*
INFO:tensorflow:Tensor: bert/encoder/layer_0/attention/self/key_p/kernel:0 *INIT_FROM_CKPT*
INFO:tensorflow:Tensor: bert/encoder/layer_0/attention/self/key_q/kernel:0 *INIT_FROM_CKPT*
```

If success, a checkpoint of the result model will be in output directory.

### 2. Finetune

Run the script `run.sh`:

```bash
export OUTPUT_DIR=~/SST-2_Pruning
export CHECKPOINT=uncased_L-12_H-768_A-12_f
export BERT_DIR=`pwd`

task_name="SST-2"
batch_size="32"
max_seq_length="128"
fine_tune_epoch="50.0"
learning_rate="0"
learning_rate_warmup="200"
lambda_lr="10.0"
alpha_lr="5.0"
target_sparsity="0.95"
target_sparsity_warmup="4000"
hidden_dropout_prob="0.1"
attention_probs_dropout_prob="0.1"
regularization_scale="0"

python ./flop/run_classifier.py \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=true \
    --data_dir=$BERT_DIR/datasets/$task_name/ \
    --vocab_file=$BERT_DIR/$CHECKPOINT/vocab.txt \
    --bert_config_file=$BERT_DIR/$CHECKPOINT/bert_config.json \
    --init_checkpoint=$BERT_DIR/$CHECKPOINT/bert_model_f.ckpt \
    --max_seq_length=$max_seq_length \
    --train_batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_train_epochs=$fine_tune_epoch \
    --learning_rate_warmup=$learning_rate_warmup \
    --lambda_learning_rate=$lambda_lr \
    --alpha_learning_rate=$alpha_lr \
    --target_sparsity=$target_sparsity \
    --hidden_dropout_prob=$hidden_dropout_prob \
    --attention_probs_dropout_prob=$attention_probs_dropout_prob \
    --target_sparsity_warmup=$target_sparsity_warmup \
    --regularization_scale=$regularization_scale \
    --output_dir=$OUTPUT_DIR/$CHECKPOINT
```

Adjust arguments if you need, more specific details please check the paper. In addition, in order to solve the problem of overfitting, I also add **l2 regularization** on dense layers.

The `output_dir` will store the checkpoints and a tensorboard's summary file. The evaluate metrics on dev set will also be summarized in that directory. 

Each training output will store in a folder named by a timestamp string. For example: `SST-2_Pruning/uncased_L-12_H-768_A-12_f/2020-06-02-12:15:59`.

#### Tensorboard Scalars

In `flop/optimization_flop.py` ,  loss, expected sparsity and each parameters' learning rates are summarized. Also, when training the model,  the program will save model's checkpoint per 1000 (parameter `save_checkpoints_steps` in `run_classifier.py`) steps, and `tf.estimator.train_and_evaluate()` evaluate new checkpoint in dev set. The evaluate result will be summarized as well. 

Suppose we select training summary and evaluate summary at the same time:

![](http://47.101.132.64:8888/images/2020/06/03/blob70b1f8508d5925df.jpg)

Following charts will be shown:

![image-20200603220345047](http://47.101.132.64:8888/images/2020/06/03/blob835bb06d7f2c2f6f.jpg)

![](http://47.101.132.64:8888/images/2020/06/03/blob2cf743a3ee3ea712.jpg)

Scalars description:

+ `lambda_1_1`, `lambda_2_1` : Two Lagrange multipliers in l0 regularization pruning.

+ `alpha_lr`, `lambda_lr`, `model_lr` : Learning rate of alphas, lambdas and BERT model parameters.
+ `expected_sparsity` , `target_sparsity`: Expected sparsity calculated by model's alphas parameters, and target sparsity which is warm up by `target_sparsity_warmup` steps.
+ `l2_regularization_loss` : Sum of all dense layers' l2 regularization value.
+ `eval_accuracy`, `precision`, `recall`, `f1_score`: Metrics on entire dev set evaluated on checkpoint.
+ `loss_1`: MSE loss in that training step.
+ `loss`: MSE loss on entire dev set evaluated on checkpoint.

#### Tensorboard Hyperparameters

All hyperparameters will be stored as summary text:

![](http://47.101.132.64:8888/images/2020/06/03/blobec168c629a2459a9.jpg)

#### Finetune Result

My finetune result:

![](http://47.101.132.64:8888/images/2020/06/07/blob.jpg)

Then we can get a checkpoint (80% sparsity, 0.9 accuracy).

### 3. Remove Mask

Run the script `remove_mask.sh`:

```bash
python ./flop/remove_mask.py \
  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
  --checkpoint=/path/to/checkpoint \
  --output_folder_dir=/path/to/output/directory
```

After running, checkpoint and config file will output to `output_folder_dir`. Parameters information will be shown in `info.txt`:

```
dense_total_params: 233570304
dense_pruned_params: 41538048
dense_origin_params: 84934656
dense_sparsity: 0.822160
non_kernel_params: 23959298
total_params: 108893954
pruned_total_params: 65497346
actual_compact_rate: 0.601478
```

### 4. Finetune Again

![](http://47.101.132.64:8888/images/2020/06/07/blob35f73e38aabe827d.jpg)

Then we get a model with 92.43% accuracy and 65M parameters, compare with BERT base:

| Parameters | SST-2  |
| ---------- | ------ |
| 108M(100%) | 93.35% |
| 65M(60%)   | 92.43% |

| Metrics\Model | BERT base | Pruned model |
| ---- | ---- | ---- |
| Checkpoint size | 421MB | 253MB (60%) |
| Memory allocation | 1399MB | 879MB (63%) |
| Latency | 8.45s | 8.92s (105%) |

(Test on SST-2 dev set, batch size=8, TITAN RTX)

[Download model here](https://drive.google.com/file/d/1jIFzyjjIoL-A8j5SQNtoJiSf1VsdfTbG/view?usp=sharing)

## Cite

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
@article{wang2019structured,
  title={Structured Pruning of Large Language Models},
  author={Wang, Ziheng and Wohlwend, Jeremy and Lei, Tao},
  journal={arXiv preprint arXiv:1910.04732},
  year={2019}
}
```

