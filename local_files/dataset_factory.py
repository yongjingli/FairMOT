from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.jde_kps import JointDatasetKps
from .dataset.jde_kpwh import JointDatasetKpWh


def get_dataset(dataset, task):
  if task == 'mot':
    return JointDataset
  elif task == 'mot_kp':
    return JointDatasetKps
  elif task == 'mot_kpwh':
    return JointDatasetKpWh
  else:
    return None    
