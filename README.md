## Software requirements:

see requirements.txt

By the way, the designated experimental environment version is relatively low. We have tested the new environment and found that this project can also run on PyTorch 2.0.1 and Python 3.11.3 CUDA 11.4.

It is recommended to install the correct Python version and torch, CUDA, and other versions first, then follow the following command to install the transformer library(necessary step), and then install the missing libraries one by one from the requirements.

## Install transformers: (Please note that special installation is required for this package. We integrated our model files into it and made some modifications. We use the transformer framework and develop in an editable way.)

pip install --editable .

## Dataset: (Put ZHIHU-16K/ in the /transformers/examples/pytorch/sequence-labeling/ directory)

download and preprocess dataset from https://anonymous.4open.science/r/ZHIHU-16K-2640/

## Model file:

/src/transformers/models/bert/modeling_bert.py

## Project path:

/transformers/examples/pytorch/sequence-labeling/

## Train:

cd .../transformers/examples/pytorch/sequence-labeling/

python -m torch.distributed.launch --nproc_per_node=8 run_language_modeling.py --output_dir=tmp/ad_zhihu_test   --model_type=bert   --model_name_or_path=bert-base-chinese   --do_train   --do_eval   --evaluate_during_training    --train_data_file=ZHIHU-16K/ZHIHU_train/   --eval_data_file=ZHIHU-16K/ZHIHU_val/  --line_by_line --block_size 32   --num_train_epochs 5   --learning_rate 5e-6   --warmup_steps 1600   --logging_steps 100   --save_steps 100   --per_device_train_batch_size 4   --gradient_accumulation_steps 1   --overwrite_output_dir --evaluation_strategy=steps --out_file_name ad_zhihu  --save_total_limit 1 --eval_label2 --con_loss --label_num 2 --add_graph_data

The following code can be removed：-m torch.distributed.launch --nproc_per_node=8
## Test:

Firstly, it is necessary to copy all files saved to the --output_dir/checkpoint/directory to the previous directory, which is --output_dir/

python test.py --model_type=bert --model_name_or_path=output_dir/  --do_eval --eval_data_file=ZHIHU-16K/ZHIHU_test0/  --line_by_line --block_size 32 --
per_device_eval_batch_size 2 --label_num 2 --out_info_file test_information-xxx.xlsx --eval_label2 --con_loss --add_graph_data

ZHIHU_test0-ZHIHU_test12 are the test results of 13 topics respectively.
