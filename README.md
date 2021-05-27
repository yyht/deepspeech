# deepspeech
deepspeech on tensorflow (1.x ) and supported for tpu, gpu

# supported model
deepspeeech2, conformer, 

# supported loss
ctc and rnnt-loss(borrowed from Tensorflow-ASR project)

# pretrained model for chinese
training on open-sourced data:

aishell, aidatatang, magcidata and stcmds totally 1300-hours
with spec-augment.

## conformer-base pretrain
conformer-base contains 8-layers and identicial parameters just like BERT.
Different to BERT, we add T5-style relative position attention.

The pretraining is based on Wave2vec2.0. The core difference is replace VQ_VAE with linear layer projection the same to (Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition).

Besides, we also replace InfoNCE with circle-loss for Mask prediction for better representation learning.
You can download from
[link](https://drive.google.com/file/d/1srnWCrrLdiR4kepB_3fBFcVlGbfAAFfY/view?usp=sharing)


## CTC-finetuning
After pretraining, we do ctc-fintuning on labeled data. The acoustics unit is chinese-syllable about 1200 syllables from 5k chinese char.

We finetune pretrained model with dense-CTC that [blank] is size(vocab)+1
for training and decoding.

The input feature is log-mel-spectrum with feature, signal normalization.

## Released Models
We are initially releasing fintuned models(evaluated on CER for syllable and transform char to syllable with pypinyin):

| Model | Layers | Hidden Size | Params | magic_data(test)| aidatatang(test)| thchs_30(test) | Download |
| --- | --- | --- | --- | ---  | --- |---  | --- |
| Conformer-base | 8 | 768 | 286M | 3.378| 7.544| 11.997  | [link](https://drive.google.com/file/d/1B_suFqxt2pWgzFeRb_CwEv-YtmWVioVW/view?usp=sharing) |

## Noting

We just use ctc-greedy-decoder since beam-search could not bring any improvments. If you apply syllable-based Ngram model, it could achieve better results.

If you want to decode to char, you will need domain-specific language model to transfor syllable to char using HMM, machine-translation-based models for real applications.


