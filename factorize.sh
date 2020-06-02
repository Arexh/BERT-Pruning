python ./flop/factorize.py \
  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
  --checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
  --output_dir=./factorized_model/bert_model_f.ckpt \
  --finetuned