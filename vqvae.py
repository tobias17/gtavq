"""
migrated to tinygrad from: https://github.com/commaai/commavq/blob/048e825079949b86b8f6ccaeee5315d846c633dd/utils/vqvae.py
which was adapted from: https://github.com/CompVis/taming-transformers
"""
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import fetch, tqdm
from tinygrad.nn.state import load_state_dict, torch_load
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, List

@dataclass
class CompressorConfig:
   in_channels: int = 3
   out_channels: int = 3
   ch_mult: tuple[int,...] = (1,1,2,2,4)
   attn_resolutions: tuple[int] = (16,)
   resolution: int = 256
   num_res_blocks: int = 2
   z_channels: int = 256
   vocab_size: int = 1024
   ch: int = 128
   dropout: float = 0.0

   @property
   def num_resolutions(self):
      return len(self.ch_mult)

   @property
   def quantized_resolution(self):
      return self.resolution // 2**(self.num_resolutions-1)

nonlinearity = Tensor.swish

def Normalize(in_channels):
   return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample:
   def __init__(self, in_channels:int):
      self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
   def __call__(self, x:Tensor) -> Tensor:
      return self.conv(x.interpolate((x.shape[-2]*2, x.shape[-1]*2), mode="nearest"))

class Downsample:
   def __init__(self, in_channels):
      # no asymmetric padding in torch conv, must do it ourselves
      self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
   def __call__(self, x:Tensor) -> Tensor:
      return self.conv(x.pad((None,None,(0,1),(0,1)))) # type: ignore

class ResnetBlock:
   def __init__(self, *, in_channels:int, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
      self.in_channels = in_channels
      out_channels = in_channels if out_channels is None else out_channels
      self.out_channels = out_channels
      self.use_conv_shortcut = conv_shortcut

      self.norm1 = Normalize(in_channels)
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      if temb_channels > 0:
         self.temb_proj = nn.Linear(temb_channels, out_channels)
      self.norm2 = Normalize(out_channels)
      self.dropout = partial(Tensor.dropout, p=dropout)
      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
      if self.in_channels != self.out_channels:
         if self.use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
         else:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

   def __call__(self, x:Tensor, temb:Optional[Tensor]) -> Tensor:
      h = x.sequential([self.norm1, nonlinearity, self.conv1])
      if temb is not None:
         h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
      h = h.sequential([self.norm2, nonlinearity, self.dropout, self.conv2])
      if self.in_channels != self.out_channels:
         if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
         else:
            x = self.nin_shortcut(x)
      return x + h

class AttnBlock:
   def __init__(self, in_channels:int):
      self.in_channels = in_channels
      self.norm = Normalize(in_channels)
      self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
      self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
      self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
      self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

   def __call__(self, x:Tensor) -> Tensor:
      h_ = x
      h_ = self.norm(h_)
      q = self.q(h_)
      k = self.k(h_)
      v = self.v(h_)

      # compute attention
      b,c,h,w = q.shape
      q = q.reshape(b,c,h*w)
      q = q.permute(0,2,1)    # b,hw,c
      k = k.reshape(b,c,h*w)  # b,c,hw
      w_ = q.matmul(k)        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
      w_ = w_.mul(int(c)**(-0.5))
      w_ = w_.softmax(axis=2)

      # attend to values
      v = v.reshape(b,c,h*w)
      w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
      h_ = v.matmul(w_)        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
      h_ = h_.reshape(b,c,h,w)

      return x + self.proj_out(h_)

class VectorQuantizer:
   def __init__(self, num_embeddings:int, embedding_dim:int):
      self._embedding_dim = embedding_dim
      self._num_embeddings = num_embeddings

      self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
      # self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

   # the encode function
   def __call__(self, inputs:Tensor) -> Tuple[Tensor,Tensor]:
      b, s, c = inputs.shape
      flat_input = inputs.rearrange('b s c -> (b s) c')

      # Calculate distances
      distances =   Tensor.sum(flat_input**2, axis=1, keepdim=True) \
                  + Tensor.sum(self._embedding.weight**2, axis=1)   \
                  - 2 * Tensor.matmul(flat_input, self._embedding.weight.T)

      # Encoding
      encoding_indices = distances.argmin(axis=1).unsqueeze(1)
      quantized = self.embed(encoding_indices)
      print(b, s, c)
      print(encoding_indices.shape)
      encoding_indices = encoding_indices.rearrange('(b s) 1 -> b s', b=b, s=s)
      return quantized, encoding_indices

   # the decode function
   def decode(self, encoding_indices:Tensor) -> Tensor:
      return self.embed(encoding_indices)

   def embed(self, encoding_indices:Tensor) -> Tensor:
      return self._embedding(encoding_indices)

@dataclass
class DownBlock:
   block: List[ResnetBlock]
   attn:  List[AttnBlock]
   downsample: Optional[Downsample] = None

@dataclass
class MidBlock:
   block_1: ResnetBlock
   attn_1:  AttnBlock
   block_2: ResnetBlock

@dataclass
class UpBlock:
   block: List[ResnetBlock]
   attn:  List[AttnBlock]
   upsample: Optional[Upsample] = None

class Encoder:
   def __init__(self, config=CompressorConfig()):
      self.config = config
      self.temb_ch = 0
      # downsampling
      self.conv_in = nn.Conv2d(self.config.in_channels, self.config.ch, kernel_size=3, stride=1, padding=1)

      curr_res = self.config.resolution
      in_ch_mult = (1,)+tuple(self.config.ch_mult)
      self.down: List[DownBlock] = []
      for i_level in range(self.config.num_resolutions):
         block = []
         attn  = []
         block_in = self.config.ch*in_ch_mult[i_level]
         block_out = self.config.ch*self.config.ch_mult[i_level]
         for _ in range(self.config.num_res_blocks):
            block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=self.config.dropout))
            block_in = block_out
            if curr_res in self.config.attn_resolutions:
               attn.append(AttnBlock(block_in))
         down = DownBlock(block, attn)
         if i_level != self.config.num_resolutions-1:
            down.downsample = Downsample(block_in)
            curr_res = curr_res // 2
         self.down.append(down)

      # middle
      self.mid = MidBlock(
         ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout),
         AttnBlock(block_in),
         ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout),
      )
      # end
      self.norm_out = Normalize(block_in)
      self.conv_out = nn.Conv2d(block_in, self.config.z_channels, kernel_size=3, stride=1, padding=1)

      # quantizer
      self.quant_conv = nn.Conv2d(self.config.z_channels, self.config.z_channels, 1)
      self.quantize = VectorQuantizer(self.config.vocab_size, self.config.z_channels)

   def __call__(self, x:Tensor) -> Tensor:
      # timestep embedding
      temb = None

      # downsampling
      hs = [self.conv_in(x)]
      for i_level in range(self.config.num_resolutions):
         for i_block in range(self.config.num_res_blocks):
            h = self.down[i_level].block[i_block](hs[-1], temb)
            if len(self.down[i_level].attn) > 0:
               h = self.down[i_level].attn[i_block](h)
            hs.append(h)
         if i_level != self.config.num_resolutions-1:
            assert (downsample := self.down[i_level].downsample) is not None
            hs.append(downsample(hs[-1]))

      # middle
      h = hs[-1]
      h = self.mid.block_1(h, temb)
      h = self.mid.attn_1(h)
      h = self.mid.block_2(h, temb)

      # end
      h = self.norm_out(h)
      h = nonlinearity(h)
      h = self.conv_out(h)

      # run the encoder part of VQ
      h = self.quant_conv(h)
      h = h.rearrange('b c h w -> b (h w) c')
      _, encoding_indices = self.quantize(h)
      return encoding_indices

   def load_from_pretrained(self, url='https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin') -> 'Encoder':
      load_state_dict(self, torch_load(str(fetch(url))))
      return self

class Decoder:
   def __init__(self, config=CompressorConfig()):
      self.temb_ch = 0
      self.config = config

      # compute in_ch_mult, block_in and curr_res at lowest res
      block_in = self.config.ch*self.config.ch_mult[self.config.num_resolutions-1]
      curr_res = self.config.quantized_resolution

      # quantizer
      self.post_quant_conv = nn.Conv2d(config.z_channels, config.z_channels, 1)
      self.quantize = VectorQuantizer(config.vocab_size, config.z_channels)

      # z to block_in
      self.conv_in = nn.Conv2d(self.config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

      # middle
      self.mid = MidBlock(
         ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout),
         AttnBlock(block_in),
         ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout),
      )

      # upsampling
      self.up: List[UpBlock] = []
      for i_level in reversed(range(self.config.num_resolutions)):
         block = []
         attn  = []
         block_out = self.config.ch*self.config.ch_mult[i_level]
         for _ in range(self.config.num_res_blocks+1):
            block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=self.config.dropout))
            block_in = block_out
            if curr_res in self.config.attn_resolutions:
               attn.append(AttnBlock(block_in))
         up = UpBlock(block, attn)
         if i_level != 0:
            up.upsample = Upsample(block_in)
            curr_res = curr_res * 2
         self.up.insert(0, up) # prepend to get consistent order

      # end
      self.norm_out = Normalize(block_in)
      self.conv_out = nn.Conv2d(block_in, self.config.out_channels, kernel_size=3, stride=1, padding=1)

   def __call__(self, encoding_indices):
      # run the decoder part of VQ
      z = self.quantize.decode(encoding_indices)
      z = z.rearrange('b (h w) c -> b c h w', w=self.config.quantized_resolution)
      z = self.post_quant_conv(z)
      self.last_z_shape = z.shape

      # timestep embedding
      temb = None

      # z to block_in
      h = self.conv_in(z)

      # middle
      h = self.mid.block_1(h, temb)
      h = self.mid.attn_1(h)
      h = self.mid.block_2(h, temb)

      # upsampling
      for i_level in reversed(range(self.config.num_resolutions)):
         for i_block in range(self.config.num_res_blocks+1):
            h = self.up[i_level].block[i_block](h, temb)
            if len(self.up[i_level].attn) > 0:
               h = self.up[i_level].attn[i_block](h)
         if i_level != 0:
            assert (upsample := self.up[i_level].upsample) is not None
            h = upsample(h)

      h = self.norm_out(h)
      h = nonlinearity(h)
      h = self.conv_out(h)

      # scale
      return ((h + 1.0) / 2.0) * 255.

   def load_from_pretrained(self, url='https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin') -> 'Decoder':
      load_state_dict(self, torch_load(str(fetch(url))))
      return self

def transpose_and_clip(x:Tensor) -> Tensor:
   return x.permute(0, 2, 3, 1).clip(0, 255).cast(dtypes.uint8)

def write_video(frames_rgb, out, fps=20):
   import cv2
   size = frames_rgb[0].shape[:2][::-1]
   video = cv2.VideoWriter(out, 0, fps, size)
   for i in range(frames_rgb.shape[0]):
      video.write(cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR))
   video.release()
   return out

if __name__ == "__main__":
   Tensor.training = False
   import numpy as np

   decoder = Decoder().load_from_pretrained()
   encoder = Encoder().load_from_pretrained()
   tokens  = np.load("tokens.npy").astype(np.int64)

   # from tinygrad import TinyJit
   # @TinyJit
   # def decode_step(t:Tensor) -> Tensor:
   #    return decoder(t).realize()

   decoded_1 = decoder(Tensor(tokens[0]).reshape(1,-1).realize()).realize()
   print(f"{decoded_1.shape=}")
   encoded_1 = encoder(decoded_1).realize()
   print(f"{encoded_1.shape=}")
   decoded_2 = decoder(encoded_1).realize()

   from PIL import Image
   for img in [decoded_1, decoded_2]:
      Image.fromarray(transpose_and_clip(img).numpy()[0]).show()

   # decoded_frames = []
   # for i in tqdm(range(120)):
   #    frame = decode_step(Tensor(tokens[i]).reshape(1,-1).realize())
   #    decoded_frames.append(transpose_and_clip(frame).realize())
   # decoded_video = Tensor.cat(*decoded_frames)

   # write_video(decoded_video.numpy(), '/tmp/decoded.avi', fps=20)
