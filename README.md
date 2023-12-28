Install: (We use the transformer framework and develop in an editable way.)
cd .../transformers/
pip install --editable .

Software requirements:
see requirements.txt

Train:
cd .../transformers/examples/pytorch/sequence-labeling/
python -m torch.distributed.launch --nproc_per_node=8 run_language_modeling.py --output_dir=tmp/ad_zhihu_test   --model_type=bert   --model_name_or_path=bert-base-chinese   --do_train   --do_eval   --evaluate_during_training    --train_data_file=ZHIHU-16K/ZHIHU_train/   --eval_data_file=ZHIHU-16K/ZHIHU_val/  --line_by_line --block_size 32   --num_train_epochs 5   --learning_rate 5e-6   --warmup_steps 1600   --logging_steps 10   --save_steps 10   --per_device_train_batch_size 4   --gradient_accumulation_steps 1   --overwrite_output_dir --evaluation_strategy=steps --out_file_name ad_zhihu  --save_total_limit 1 --eval_label2 --con_loss --label_num 2

Test:
python test.py --model_type=bert   --output_dir=tmp/ad_zhihu/ --model_name_or_path=tmp/xxx/  --do_eval --eval_data_file=ZHIHU-16K/ZHIHU_test0/  --line_by_line --block_size 32 --per_device_eval_batch_size 2 --label_num 2 --out_info_file test_information-xxx.xlsx --eval_label2 --con_loss --add_graph_data --gcn_hidden_size 1547
