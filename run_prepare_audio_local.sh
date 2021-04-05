python prepare_audio_tfrecord.py \
	--input_path  \
	--input_meta_path \
	--input_transcript_path \
	--input_speaker_meta_path asr/metadata/SPKINFO.txt\
	--noise_path asr/ \
	--noise_meta_path asr/MS_SNSD/noise_meta.txt \
	--output_path asr/oslr_8000 \
	--sample_rate 16000 \
	--target_sample_rate 8000