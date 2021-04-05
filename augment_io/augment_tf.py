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
from augment_io.spec_aug_tf import TFFreqMasking, TFTimeMasking

TFAUGMENTATIONS = {
  "freq_masking": TFFreqMasking,
  "time_masking": TFTimeMasking,
}


class TFAugmentationExecutor:
  def __init__(self, augmentations):
    self.augmentations = augmentations

  def augment(self, inputs):
    outputs = inputs
    for au in self.augmentations:
      outputs = au.augment(outputs)
    return outputs


class Augmentation:
  def __init__(self, config = None, use_tf = False):
    if not config: config = {}
    self.before = self.parse(config.pop("before", {}))
    self.after = self.parse(config.pop("after", {}))

  @staticmethod
  def parse(config):
    augmentations = []
    for key, value in config.items():
      au = TFAUGMENTATIONS.get(key, None)
      if au is None:
        continue
      aug = au(**value) if value is not None else au()
      augmentations.append(aug)
    return TFAugmentationExecutor(augmentations)
