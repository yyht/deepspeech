# -*- coding: utf-8 -*-

import re
import jieba_fast as jieba
from pypinyin import pinyin, lazy_pinyin, Style

CH_PUNCTUATION = u"['\\\\,\\!#$%&\'()*+-/:￥;<=>.?\\n@[\\]^▽_`{|}~'－＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")
from collections import OrderedDict

non_ch = u"[^0-9a-zA-Z一-龥㖏\\s\t]+"

num_en = u"[0-9a-z]+"

def char2pinyin(input_text):
  utt = re.sub("(\\[fil\\])", "", input_text)
  utt = re.sub("(\\[spk\\])", "", utt)

  utt = re.sub(non_ch, "", utt)
  cutted_lst = list(jieba.cut(utt))
  token_pinyin_lst = []
  skip_flag = False
  for item in cutted_lst:
    if re.findall(num_en, item):
      skip_flag = True
      continue
    if skip_flag:
      break
    pinyin_lst = lazy_pinyin(item, style=Style.TONE3, neutral_tone_with_five=True)
    token_pinyin_lst.extend(pinyin_lst)
  return token_pinyin_lst
    