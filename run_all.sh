export BERT_BASE_DIR=/path/to/bert_base_pretain_model
export GLUE_DIR=/path/to/glue_dataset
export DATA_DIR=/path/to/output_dir

tasks=( "RTE" "STS-B" "WNLI" "SST-2" "MRPC" "QNLI" "QQP" "MNLI" )

batch_size="32"
max_seq_length="128"
fine_tune_epoch="3.0"
learning_rates=( "5e-5" "4e-5" "3e-5" "2e-5" )

cd bert

for task_name in ${tasks[@]}
do
  for learning_rate in ${learning_rates[@]}
  do
    python run_classifier.py \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/$task_name/ \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=$max_seq_length \
    --train_batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_train_epochs=$fine_tune_epoch \
    --output_dir=$DATA_DIR/fine_tune_outputs/$task_name/lr_$learning_rate
  done
done