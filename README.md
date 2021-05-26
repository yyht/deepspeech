# deepspeech
deepspeech on tensorflow (1.x ) and supported for tpu, gpu

# supported model
deepspeeech2, conformer, 

# supported loss
ctc and rnnt-loss(borrowed from Tensorflow-ASR project)

# pretrained model for chinese
training on open-sourced data:

aishell, aidatatang, magcidata and stcmds totally 1300-hours.

## conformer-base pretrain
conformer-base contains 8-layers and identicial parameters just like BERT.
Different to BERT, we add T5-style relative position attention.

The pretraining is based on Wave2vec2.0. The core difference is replace VQ_VAE with linear layer projection the same to (Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition)

Besides, we also replace InfoNCE with circle-loss for Mask prediction for better representation learning.

## CTC-finetuning
After pretraining, we do ctc-fintuning on labeled data. The acoustics unit is chinese-syllable about 1200 syllables from 5k chinese char.

## Released Models
We are initially releasing fintuned models:

| Model | Layers | Hidden Size | Params | magic_data| aidatatang| thchs_30 | Download |
| --- | --- | --- | --- | ---  | --- |---  | --- |
| Conformer-base | 8 | 768 | 286M | 3.378| 7.544| 11.997  | [link](https://drive.google.com/file/d/1B_suFqxt2pWgzFeRb_CwEv-YtmWVioVW/view?usp=sharing) |


