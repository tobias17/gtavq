from datasets import load_dataset # type: ignore
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.helpers import tqdm, fetch, Context
from tinygrad.nn.state import load_state_dict, torch_load, get_state_dict, get_parameters
from examples.stable_diffusion import StableDiffusion
from vqvae import Decoder
from PIL import Image
import numpy as np
import os

START_AFTER = -1
BATCH_SIZE = 20
AMOUNT_PER = 240

DATASET_ROOT  = "/raid/datasets/depthvq"
DEPTHMAP_ROOT = f"{DATASET_ROOT}/depthmaps"
LATENTS_ROOT  = f"{DATASET_ROOT}/latents"

def main():
   assert os.path.exists(DEPTHMAP_ROOT)
   if not os.path.exists(LATENTS_ROOT):
      os.mkdir(LATENTS_ROOT)

   GPUS = tuple([f"{Device.DEFAULT}:{i}" for i in range(6)])

   model = StableDiffusion()
   load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False) # type: ignore
   for k, w in get_state_dict(model).items():
      if "first_stage_model" in k:
         w.replace(w.shard(GPUS).realize())

   decoder = Decoder().load_from_pretrained()
   for w in get_parameters(decoder):
      w.replace(w.shard(GPUS).realize())

   @TinyJit
   def encode_latent(im:Tensor) -> Tensor:
      x = im.div(255.0) * 2.0 - 1.0
      z = model.first_stage_model.quant_conv(model.first_stage_model.encoder(x)).chunk(2, dim=1)[0]
      z = z.mul(0.18215).cast(dtypes.float16).rearrange('b c h w -> b (h w) c')
      return z.realize()

   dataset = load_dataset("/raid/datasets/commavq", trust_remote_code=True)
   for split_key, split in dataset.items():
      if START_AFTER >= 0 and split_key in [f"{i}" for i in range(START_AFTER)]:
         continue
      print(f"Starting split '{split_key}'")
      for filepath in tqdm(split["path"]):
         filename = os.path.basename(filepath)
         scene_folder = filename.split(".")[0]
         all_tokens = None

         start_i = 0
         while True:
            latents_path = os.path.join(LATENTS_ROOT, split_key, scene_folder, f"latent_{start_i}.npz")
            latents_dirpath = os.path.dirname(latents_path)
            if not os.path.exists(latents_path) or len(os.listdir(latents_dirpath)) < (AMOUNT_PER // BATCH_SIZE):
               depthmaps = []
               for frame_i in range(AMOUNT_PER):
                  depthmap_path = os.path.join(DEPTHMAP_ROOT, split_key, scene_folder, f"{start_i+frame_i:04d}.png")
                  if not os.path.exists(depthmap_path):
                     break
                  try:
                     img_gs = Image.open(depthmap_path).convert('RGB')
                     img_rgb = Image.new('RGB', img_gs.size)
                     img_rgb.paste(img_gs)

                     im = Tensor(np.array(img_rgb)).cast(dtypes.float32).unsqueeze(0).rearrange('b h w c -> b c h w')
                     depthmaps.append(im.realize())
                  except Exception as ex:
                     print(f"Ran into error loading {depthmap_path}: {ex}")
                     break
               else:
                  if all_tokens is None:
                     all_tokens = np.load(filepath)
                  assert isinstance(all_tokens, np.ndarray)
                  tokens = Tensor(all_tokens[start_i:start_i+AMOUNT_PER]).rearrange('b h w -> b (h w)').shard(GPUS, axis=0).realize()

                  depths = Tensor.cat(*depthmaps).shard(GPUS, axis=0).realize()
                  # assert depths.shape == frames.shape, f"shape mismatch, {depths.shape} != {frames.shape}"

                  with Context(BEAM=1):
                     frames = decoder(tokens).realize()
                     frames_z = encode_latent(frames.realize()).numpy()
                     depths_z = encode_latent(depths.realize()).numpy()
                  assert depths_z.shape == frames_z.shape, f"shape mismatch, {depths_z.shape} != {frames_z.shape}"

                  os.makedirs(latents_dirpath, exist_ok=True)
                  for slice_i in range(0, AMOUNT_PER, BATCH_SIZE):
                     latents_filename = os.path.join(LATENTS_ROOT, split_key, scene_folder, f"latent_{slice_i}.npz")
                     frames_slice = frames_z[slice_i:slice_i+BATCH_SIZE]
                     depths_slice = depths_z[slice_i:slice_i+BATCH_SIZE]
                     assert frames_slice.shape[0] == BATCH_SIZE and depths_slice.shape[0] == BATCH_SIZE
                     np.savez(latents_filename, frames=frames_slice, depths=depths_slice)

            start_i += AMOUNT_PER
            if start_i >= AMOUNT_PER:
               break

if __name__ == "__main__":
   main()
