from tinygrad import Tensor, nn

class LatentGptConfig:
   max_context: int = 16
   max_size: int = 128
   in_dims: int = 8
   out_dims: int = 4
   n_layer: int = 16
   n_head: int = 16
   dim: int = 1024
   ff_mult: float = 3.0

class Attention:
   def __init__(self, config:LatentGptConfig, is_causal:bool):
      assert config.dim % config.n_head == 0
      self.config = config
      self.is_causal = is_causal

      self.to_q = nn.Linear(config.dim, config.dim)
      self.to_k = nn.Linear(config.dim, config.dim)
      self.to_v = nn.Linear(config.dim, config.dim)
      self.out  = nn.Linear(config.dim, config.dim)

   def __call__(self, x:Tensor) -> Tensor:
      q,k,v = self.to_q(x), self.to_k(x), self.to_v(x)
      q,k,v = [y.rearrange('b s1 s2 (n d) -> b s1 n s2 d', n=self.config.n_head) for y in (q,k,v)]

      y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
      y = y.rearrange('b s1 n s2 d -> b s1 s2 (n d)')

      return self.out(y)

class FeedForward:
   def __init__(self, config:LatentGptConfig):
      hidden_dim = int(config.dim * config.ff_mult)
      self.proj_in  = nn.Linear(config.dim, hidden_dim)
      self.proj_out = nn.Linear(hidden_dim, config.dim)
   
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.proj_in, Tensor.quick_gelu, self.proj_out])

class TransformerBlock:
   def __init__(self, config:LatentGptConfig):
      self.t_attn = Attention(config, is_causal=True)
      self.f_attn = Attention(config, is_causal=False)
      self.mlp    = FeedForward(config)
      self.ln_1   = nn.LayerNorm(config.dim)
      self.ln_2   = nn.LayerNorm(config.dim)
      self.ln_3   = nn.LayerNorm(config.dim)
   
   def __call__(self, x:Tensor) -> Tensor:
      h = x.rearrange('b t f c -> b f t c')
      h = h + self.t_attn(self.ln_1(h))
      h = h.rearrange('b f t c -> b t f c')
      h = h + self.f_attn(self.ln_2(h))
      h = h + self.mlp(self.ln_3(h))
      return h

class GPT:
   def __init__(self, config=LatentGptConfig()):
      self.config = config
      self.proj_in = nn.Linear(config.in_dims, config.dim)
      self.proj_out = nn.Linear(config.dim, config.out_dims)
      self.pos_c_embed = nn.Embedding(config.max_context, config.dim)
      self.pos_s_embed = nn.Embedding(config.max_size, config.dim)
      self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
      self.ln_f = nn.LayerNorm(config.dim)

   def __call__(self, frames:Tensor, depths:Tensor) -> Tensor:
      assert len(frames.shape) == 4
      assert frames.shape[2:] == (512,4)
      assert frames.shape == depths.shape

      x = frames.cat(depths, dim=-1)
      pos_c = self.pos_c_embed(Tensor.arange(frames.shape[1], device=frames.device).reshape(1,-1,1,1))
      pos_s = self.pos_s_embed(Tensor.arange(frames.shape[2], device=frames.device).reshape(1,1,-1,1))
      x = self.proj_in(x) + pos_c + pos_s
      x = x.sequential(self.layers) # type: ignore
      x = self.proj_out(self.ln_f(x))
      return x
