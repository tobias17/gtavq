import numpy as np
import os

DATASET_ROOT = "/raid/datasets/depthvq/batched"

for root, _, files in os.walk(DATASET_ROOT):
   for filename in files:
      if filename.endswith(".npz"):
         path = os.path.join(root, filename)
         print(path)
         np.load(path)
