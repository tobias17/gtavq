from datasets import load_dataset # type: ignore
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import tqdm
from vqvae import Decoder
from PIL import Image
from threading import Thread

import sys, os
sys.path.append("../depth-fm")
from depthfm import DepthFM
import torch

BATCH_SIZE = 24

def async_save(depth:np.ndarray, i:int, out_root:str):
   for j in range(BATCH_SIZE):
      Image.fromarray(depth[j]).save(f"{out_root}/{i+j:04d}.png")

def main():
   dataset = load_dataset("/home/tobi/datasets/commavq", trust_remote_code=True)
   decoder = Decoder().load_from_pretrained()

   model = DepthFM("../depth-fm/checkpoints/depthfm-v1.ckpt")
   model.cuda().eval()
   model.model.dtype = torch.float16

   @TinyJit
   def decode_step(t:Tensor) -> Tensor:
      return decoder(t).realize()

   for split_key, split in dataset.items():
      print((banner := "\n"+"="*80+"\n\n") + f"Starting split '{split_key}'" + banner[::-1])
      for filepath in split["path"]:
         out_root = f"./depthmaps/{split_key}/{os.path.basename(filepath).split('.')[0]}"
         if not os.path.exists(out_root):
            os.makedirs(out_root)
         print(out_root)

         tokens = np.load(filepath).astype(np.int64)
         assert tokens.shape[0] % BATCH_SIZE == 0
         for i in tqdm(range(0, tokens.shape[0], BATCH_SIZE)):
            t = Tensor(tokens[i:i+BATCH_SIZE]).reshape(BATCH_SIZE, -1).realize()
            frames = decode_step(t)
            im = torch.from_numpy((frames.clip(0, 255) / 127.5 - 1).numpy().astype(np.float16)).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.half):
               depth = model.predict_depth(im, num_steps=1, ensemble_size=0)
            depth = depth.squeeze(1).cpu().numpy()

            depth = (depth * 255).astype(np.uint8)
            Thread(target=async_save, args=(depth,i,out_root)).start()

if __name__ == "__main__":
   main()
