from tinygrad import Tensor, nn

class DepthGptConfig:
   max_context: int = 20
   frame_vocab: int = 1024
   depth_vocab: int = 256
   n_layer: int = 12
   n_head: int = 16
   dim: int = 1024
   ff_mult: float = 4.0

class Attention:
   def __init__(self, config:DepthGptConfig, is_causal:bool):
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
   def __init__(self, config:DepthGptConfig):
      hidden_dim = int(config.dim * config.ff_mult)
      self.proj_in  = nn.Linear(config.dim, hidden_dim)
      self.proj_out = nn.Linear(hidden_dim, config.dim)
   
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.proj_in, Tensor.quick_gelu, self.proj_out])

class TransformerBlock:
   def __init__(self, config:DepthGptConfig):
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
   def __init__(self, config=DepthGptConfig()):
      self.config = config
      self.frame_embed = nn.Embedding(config.frame_vocab)
      self.depth_embed = nn.Embedding(config.depth_vocab)
      self.time_embed  = nn.Embedding(config.max_context)
      self.layers      = [TransformerBlock(config) for _ in range(config.n_layer)]
      self.ln_f        = nn.LayerNorm(config.dim)
      self.lm_head     = nn.Linear(config.dim, config.frame_vocab, bias=False)

   def __call__(self, frames:Tensor, depths:Tensor) -> Tensor:
      assert frames.shape == depths.shape and len(frames.shape) == 3
      x = self.frame_embed(frames) + self.depth_embed(depths) + self.time_embed(Tensor.arange(frames.shape[1]))
      x = x.sequential(self.layers)
      x = self.ln_f(x)
      return self.lm_head(x)
