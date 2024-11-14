import os
import numpy as np
from PIL import Image
from create_depthmaps import get_filepath_for
from tinygrad.helpers import trange

DS_ROOT  = f"/net/tiny/raid/datasets/depthvq"
IN_ROOT  = f"{DS_ROOT}/depthmaps"
OUT_ROOT = f"{DS_ROOT}/depthpacks"
assert os.path.exists(os.path.dirname(OUT_ROOT))

AMOUNT_PER = 240

def main():
   for splitdir in sorted(os.listdir(IN_ROOT)):
      for scenedir in sorted(os.listdir(f"{IN_ROOT}/{splitdir}")):
         try:
            out_folder = f"{OUT_ROOT}/{splitdir}"
            out_file = f"{out_folder}/{scenedir}.npy"
            if not os.path.exists(out_folder):
               os.makedirs(out_folder)
            elif os.path.exists(out_file):
               continue

            in_folder = f"{IN_ROOT}/{splitdir}/{scenedir}"
            if len(os.listdir(in_folder)) >= AMOUNT_PER:
               frames = []
               for i in trange(AMOUNT_PER):
                  im = np.array(Image.open(get_filepath_for(in_folder, i)))
                  assert len(im.shape) == 2
                  im = im.reshape((1,1,im.shape[0],im.shape[1]))
                  frames.append(im)
               data = np.concatenate(frames, axis=0)
               np.save(out_file, data)
         except Exception as ex:
            print(ex)

if __name__ == "__main__":
   main()
