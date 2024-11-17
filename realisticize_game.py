from tinygrad import Tensor, dtypes, TinyJit
from record_screen import capture_window
from vqvae import Encoder, Decoder, transpose_and_clip
from PIL import Image
import numpy as np
import cv2

def main():
   Tensor.training = False
   enc = Encoder().load_from_pretrained()
   dec = Decoder().load_from_pretrained()

   @TinyJit
   def run_frame(x:Tensor) -> Tensor:
      tokens = enc(x).argmax(axis=-1)
      return transpose_and_clip(dec(tokens)).squeeze(0).realize()

   while True:
      img_in = np.array(capture_window("Grand Theft Auto V"))
      img_in = img_in[40:-5, 5:-5]

      h, w, _ = img_in.shape
      target_height = w // 2
      assert target_height <= h, f"{target_height} > {h}"
      y_start = (h - target_height) // 2
      img_in = img_in[y_start:y_start+target_height]
      img_in = Image.fromarray(img_in).resize((256,128))
      
      frame_in  = Tensor(np.array(img_in)).permute(2, 0, 1).cast(dtypes.float32).unsqueeze(0)
      frame_out = run_frame(frame_in.realize())

      img_out = cv2.cvtColor(frame_out.numpy(), cv2.COLOR_RGB2BGR)
      img_out = cv2.resize(img_out, (2048,1024))
      cv2.imshow("GTA V", img_out)
      cv2.waitKey(1)

if __name__ == "__main__":
   main()
