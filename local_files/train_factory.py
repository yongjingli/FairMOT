from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .mot import MotKpTrainer
from .mot import MotKpWhTrainer


train_factory = {
  'mot': MotTrainer,
  'mot_kp': MotKpTrainer,
  'mot_kpwh': MotKpWhTrainer,
} 
