nohup python3 ./run_wav2vec_conformer_ctc_tpu.py \
	--buckets gs://yyht_source/pretrain \
	--data_dir chinese_asr_v1/ \
	--bert_config_file ./config/conformer_pretrain_v2_char.json \
	--train_file chinese_asr_v1/chinese_asr_v1_pretrain_file_list.txt \
	--output_dir chinese_asr_v1/conformer_v2_linear_ctc_char_fc_latest \
	--init_checkpoint chinese_asr_v1/conformer_pretrain_v2_linear/model.ckpt-500000 \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 128 \
	--learning_rate 5e-5 \
	--num_train_steps 500000 \
	--num_warmup_steps 20000 \
	--save_checkpoints_steps 1000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert0 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--monitoring True \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--max_duration 20 \
	--samples_per_second 8000 \
	--circle_margin 0.25 \
	--circle_gamma 32 \
	--audio_featurizer_config_path chinese_asr_v1/audio_featurizer_config.json \
	--featurizer_aug_config_path chinese_asr_v1/featurizer_aug_config.json \
	--target_feature_mode linear \
	--monitoring true \
	--transcript_seq_length 81 \
	--blank_index "-1" \
	--ctc_loss_type "dense_ctc" \
	--output_mode "char" \
	--optimizer_type "adam_decay" \
	--decoder_type "fc"
