# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import nlpaug.flow as naf

from augment_io.signal_aug_np import SignalCropping, SignalLoudness, SignalMask, SignalNoise, \
  SignalPitch, SignalShift, SignalSpeed, SignalVtlp
from augment_io.spec_aug_np import FreqMasking, TimeMasking


AUGMENTATIONS = {
  "freq_masking": FreqMasking,
  "time_masking": TimeMasking,
  "noise": SignalNoise,
  "masking": SignalMask,
  "cropping": SignalCropping,
  "loudness": SignalLoudness,
  "pitch": SignalPitch,
  "shift": SignalShift,
  "speed": SignalSpeed,
  "vtlp": SignalVtlp
}

class Augmentation:
  def __init__(self, config = None, use_tf = False):
    if not config: config = {}
    
    self.before = self.parse(config.pop("before", {}))
    self.after = self.parse(config.pop("after", {}))

  @staticmethod
  def parse(config):
    augmentations = []
    for key, value in config.items():
      au = AUGMENTATIONS.get(key, None)
      if au is None:
        continue
      aug = au(**value) if value is not None else au()
      augmentations.append(aug)
    return naf.Sometimes(augmentations)
