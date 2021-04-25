nohup python3 ./run_wav2vec_conformer_bert_base.py \
	--buckets gs://yyht_source/pretrain \
	--data_dir chinese_asr_v1/ \
	--bert_config_file ./config/conformer_pretrain_v1.json \
	--bert_lm_config ./config/bert_base_relative_t5_config.json \
	--train_file chinese_asr_v1/chinese_asr_v1_pretrain_file_list.txt \
	--output_dir chinese_asr_v1/conformer_pretrain_v2_bert \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 128 \
	--learning_rate 1e-4 \
	--num_train_steps 500000 \
	--num_warmup_steps 20000 \
	--init_checkpoint chinese_asr_v1/conformer_pretrain_v2_linear/model.ckpt-232000 \
	--bert_lm_init_checkpoint models/bert_base_relative_t5_sinusoidal_50g_official/model.ckpt-1000000 \
	--save_checkpoints_steps 1000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert1 \
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
	--transcript_seq_length 128 \
	--blank_index "0" \
	--ctc_loss_type "dense_ctc" \
	--output_mode "char" \
	--tune_mode "am" \
	--is_pretraining True \
	--optimizer_type "adafactor"
