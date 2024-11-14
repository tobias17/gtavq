from datasets import load_dataset # type: ignore
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import tqdm
from vqvae import Decoder
from PIL import Image
from threading import Thread
import time

START_AFTER = 10
BATCH_SIZE = 24
MAX_PER = BATCH_SIZE * 10

# OUT_ROOT = f"/net/tiny/raid/datasets/depthvq"
OUT_ROOT = "."

def get_filepath_for(out_root:str, index:int) -> str:
   return f"{out_root}/{index:04d}.png"

def async_save(depth:np.ndarray, out_root:str, i:int):
   for j in range(BATCH_SIZE):
      Image.fromarray(depth[j]).save(get_filepath_for(out_root, i+j))

def main():
   import sys, os
   sys.path.append("../depth-fm")
   from depthfm import DepthFM # type: ignore
   import torch

   dataset = load_dataset("/home/tobi/datasets/commavq", trust_remote_code=True)
   decoder = Decoder().load_from_pretrained()

   model = DepthFM("../depth-fm/checkpoints/depthfm-v1.ckpt")
   model.cuda().eval()
   model.model.dtype = torch.float16

   @TinyJit
   def decode_step(t:Tensor) -> Tensor:
      return decoder(t).realize()

   for split_key, split in dataset.items():
      if START_AFTER >= 0 and split_key in [f"{i}" for i in range(START_AFTER+1)]:
         continue
      print((banner := "\n"+"="*80+"\n\n") + f"Starting split '{split_key}'" + banner[::-1])
      for filepath in split["path"]:
         tokens = np.load(filepath).astype(np.int64)
         max_amount = MAX_PER if MAX_PER > 0 else tokens.shape[0]

         out_root = f"{OUT_ROOT}/depthmaps/{split_key}/{os.path.basename(filepath).split('.')[0]}"
         if not os.path.exists(out_root):
            os.makedirs(out_root)
         print(out_root)
         if len(os.listdir(out_root)) >= max_amount:
            continue

         assert max_amount % BATCH_SIZE == 0
         for i in tqdm(range(0, max_amount, BATCH_SIZE)):
            if os.path.exists(get_filepath_for(out_root, i + BATCH_SIZE - 1)):
               time.sleep(0.02)
               continue

            t = Tensor(tokens[i:i+BATCH_SIZE]).reshape(BATCH_SIZE, -1).realize()
            frames = decode_step(t)
            im = torch.from_numpy((frames.clip(0, 255) / 127.5 - 1).numpy().astype(np.float16)).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.half):
               depth = model.predict_depth(im, num_steps=1, ensemble_size=0)
            depth = depth.squeeze(1).cpu().numpy()

            depth = (depth * 255).astype(np.uint8)
            Thread(target=async_save, args=(depth,out_root,i)).start()

if __name__ == "__main__":
   main()
