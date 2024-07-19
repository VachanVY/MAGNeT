import dataclasses as dc
import inspect
import typing as tp
import math

import torch
from torch import (
    nn, Tensor
)
from torch.nn import functional as F

from .music_bench import QCODING_LEN
from .utils import sample_tokens


@dc.dataclass
class monfig: # model config
    d_model:int = 512
    num_heads:int = 8
    assert d_model % num_heads == 0
    assert d_model % 2 == 0
    dropout_rate:float = 0.0
    num_layers:int = 8
    out_norm:bool = True
    attn_window:int = 5

    cardinality:int = 1024 # sorta like vocab_size
    num_codebooks:int = 4
    max_seq_len:int = QCODING_LEN

    spanlen:int = 3


def get_sinusoidal_positional_embeddings(maxlen:int, dim:int) -> Tensor:
    import numpy as np
    p, i = np.meshgrid(np.arange(float(maxlen)), np.arange(dim/2.)*2)
    theta = (p/1e4**(i/dim)).T

    pos_emb = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    pos_emb = pos_emb.reshape((maxlen, dim))[None] # (B=1, maxlen, dim)
    return torch.from_numpy(pos_emb)

# from xformers.ops.fmha.attn_bias.LocalAttentionFromBottomRightMask
def magnet_restricted_att_mask(shape:tuple, windows:tuple, dtype:torch.dtype=torch.float32):
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    mask = torch.full(
        shape, dtype=create_as, fill_value=1
    )

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    mask = torch.triu(mask, diagonal=shift - windows[0])
    mask = torch.tril(mask, diagonal=shift + windows[1])
    mask = torch.log(mask)
    return mask.to(dtype)


# https://github.com/facebookresearch/audiocraft/blob/adf0b04a4452f171970028fcf80f101dd5e26e19/audiocraft/modules/transformer.py#L138
class Attention(nn.Module):
    def __init__(self, config:monfig):
        super().__init__()
        self.wq = nn.LazyLinear(config.d_model) # commented a warning in the source code for Lazy Modules
        self.wk = nn.LazyLinear(config.d_model)
        self.wv = nn.LazyLinear(config.d_model)

        self.w = nn.Linear(config.d_model, config.d_model)

        self.num_heads = config.num_heads
        self.hdim = config.d_model // config.num_heads
        self.dropout_rate = config.dropout_rate

    def forward(
        self,     # CROSS ATTN       SELF ATTN
        q:Tensor, # (B, T, d_model); (B, T, d_model)
        k:Tensor, # (B, N, dim)    ; (B, T, d_model)
        v:Tensor, # (B, N, dim)    ; (B, T, d_model)
        # dim: dimention from conditioning tensor
        attn_mask:tp.Optional[Tensor]=None # ONLY FOR SELF ATTENTION
        # (T, T) A float mask of the same type as query, key, value that is added to the attention score.
    ):
        T, N = q.shape[1], k.shape[1]
        q, k, v = self.wq(q), self.wk(k), self.wv(v) # (B, T, d_model), (B, N, d_model), (B, N, d_model)

        q = q.view(-1, T, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, T, hdim)
        k = k.view(-1, N, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, N, hdim)
        v = v.view(-1, N, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, N, hdim)

        # Flash Attn Shapes: q: (B, ..., T, hdim); k: (B, ..., N, hdim); v: (B, ..., N, hdim) => (B, ..., T, hdim)
        att_out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0
        ) # (B, num_heads, T, hdim)
        att_out = att_out.transpose(1, 2).contiguous().view(-1, T, self.hdim*self.num_heads) # (B, num_heads, T, hdim) => (B, T, d_model)
        
        linear_att_out = self.w(att_out) # (B, T, d_model)
        return linear_att_out
                

# https://github.com/facebookresearch/audiocraft/blob/adf0b04a4452f171970028fcf80f101dd5e26e19/audiocraft/modules/transformer.py#L454
class TransformerLayer(nn.Module):
    """Not Causal Decoder Transformer"""
    def __init__(self, config:monfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.satt = Attention(config)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.catt = Attention(config)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.norm3 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential( # ffn like in torch.nn.TransformerEncoderLayer
            nn.Linear(config.d_model, config.d_model*4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_model*4, config.d_model),
            nn.Dropout(config.dropout_rate)
        )

    def forward(
        self,
        xrc:Tensor, # (B, T, d_model)
        conditioning_tensor:tp.Optional[Tensor]=None, # (B, N, cond_dim)
        xrc_att_mask:tp.Optional[Tensor]=None, # (T, T) # FOR SELF ATTN ONLY, NOT FOR CROSS ATTN
        # cond_att_mask:tp.Optional[Tensor]=None
    ):
        # First SubBlock
        sattn_in = self.norm1(xrc)
        xrc = xrc + self.dropout1(self.satt(sattn_in, sattn_in, sattn_in, xrc_att_mask)) # (B, T, d_model)

        # Second SubBlock
        if conditioning_tensor is not None:
            xrc = xrc + self.dropout2(self.catt(self.norm2(xrc), conditioning_tensor, conditioning_tensor, None)) # (B, T, d_model)

        # Third SubBlock
        xrc = xrc + self.ffn(self.norm3(xrc)) # (B, T, d_model)
        return xrc

# https://github.com/facebookresearch/audiocraft/blob/adf0b04a4452f171970028fcf80f101dd5e26e19/audiocraft/modules/transformer.py#L577
class Transformer(nn.Module):
    def __init__(self, config:monfig):
        super().__init__()
        self.register_buffer(
            "pos_emebddings", 
            get_sinusoidal_positional_embeddings(config.max_seq_len, config.d_model) # (B=1, T, d_model)
        )

        self.blocks = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.out_norm = None
        if config.out_norm:
            self.out_norm = nn.LayerNorm(config.d_model)
        
        
    def forward(
        self,
        xrc:Tensor, # (B, T, d_model)
        conditioning_tensor:Tensor, # (B, N, cond_dim)
        xrc_att_mask:Tensor, # (T, T) # FOR SELF ATTN ONLY
    ):
        xrc += self.pos_emebddings
        for block in self.blocks:
            xrc = block(xrc, conditioning_tensor, xrc_att_mask)
        if self.out_norm is not None:
            xrc = self.out_norm(xrc)
        return xrc
        

class MAGNET(nn.Module):
    """```
    Args: 
        model: transformer model
        config: repo config
    Input:
        x:Tensor, # (B, nq, T)
        conditioning_tensor:Tensor, # (B, N, cond_dim)
        seq_mask:Tensor, # (B, T)
        cross_att_mask:Tensor # (T, N)
    ```"""
    def __init__(
        self,
        model:Transformer,
        config:monfig # Model config
    ):
        super().__init__()
        self.maxlen = config.max_seq_len
        self.nq = config.num_codebooks # nq
        self.cardinality = config.cardinality
        self.spanlen = config.spanlen

        self.register_buffer("restricted_att_mask", magnet_restricted_att_mask(
                shape=(self.maxlen, self.maxlen),
                windows=(config.attn_window, config.attn_window)
            )
        )
        self.model = model
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                # something like vocab_size
                num_embeddings=config.cardinality + 1, # +1 for mask_id
                embedding_dim=config.d_model
            ) for _ in range(self.nq)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(
                config.d_model, config.cardinality
            ) for _ in range(self.nq)
        ])

    def forward(
        self,
        x:Tensor, # (B, Nq, T)
        conditioning_tensor:tp.Optional[Tensor], # (B, N, cond_dim)
        # seq_mask:tp.Optional[Tensor]=None, # (B, T) # were doing this manually in the training loop...
        # cross_att_mask:tp.Optional[Tensor]=None # (T, N)
    ):
        T = x.shape[-1]
        # emb: sum((cardinality {in dimension T}, d_model) ==x[:, codebook]:=(B, T)=> (B, T, d_model)s)
        x = sum([self.embeddings[codebook](x[:, codebook]) for codebook in range(self.nq)]) # (B, T, d_model)
        x = self.model(x, conditioning_tensor, self.restricted_att_mask[:T, :T]) #, seq_mask, cross_att_mask) # (B, T, d_model)
        # stack((B, T, cardinality), dim=1) => (B, nq, T, cardinality)
        x = torch.stack([self.linears[codebook](x) for codebook in range(self.nq)], dim=1)
        return x # (B, nq, T, cardinality)
    
    def configure_optimizers(
        self,
        weight_decay:float,
        learning_rate:float,
        betas:tuple[float, float],
        device_type:str
    ):
        params_dict = {pname:p for pname, p in self.named_parameters() if p.requires_grad}

        # all weights except layernorms and biases, embeddings and linears
        decay_params = [p for pname, p in params_dict.items() if p.dim() >= 2]
        # layernorms and biases
        non_decay_params = [p for pname, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        other_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=learning_rate,
            betas=betas,
            **other_args
        )
        return optimizer
    
    @property
    def mask_id(self): return self.cardinality
       
    @torch.inference_mode()
    def generate(
        self,
        prompt:str|list[str],
        get_cond_tokens_func:tp.Callable[[str|list[str]], dict[str, Tensor]],
        get_cond_tensors_func:tp.Callable[[dict[str, Tensor]], Tensor],
        device:torch.device,
        init_audio_tokens:tp.Optional[Tensor]=None,
        top_k:tp.Optional[int]=None,
        top_p:tp.Optional[float]=None,
        decoding_steps:list[int] = [20, 10, 10, 10]
    ) -> Tensor:
        # generate in eval mode
        assert len(decoding_steps) == self.nq, "decoding_steps should be equal to nq"
        assert not self.training, "generation should be in eval mode"
        assert prompt is not None, "prompt shouldn't be None... until CFG is implemented"
        init_audio_len = 0
        if init_audio_tokens is not None and isinstance(init_audio_tokens, Tensor):
            init_audio_tokens = init_audio_tokens.to(device)
            init_audio_len = init_audio_tokens.shape[-1]
            assert init_audio_tokens.shape[-2] == self.nq and init_audio_len < self.maxlen

        tok_prompt = get_cond_tokens_func(
            [prompt] if isinstance(prompt, str) else prompt
        ) # (B, N)
        B = tok_prompt["input_ids"].shape[0]
        conditioning_tensor = get_cond_tensors_func(padded_cond_seq=tok_prompt) # (B, N, D=512)
        
        audio_tokens = torch.full((B, self.nq, self.maxlen), fill_value=self.mask_id).to(device) # (B, Nq, T)
        if init_audio_tokens is not None:
            audio_tokens[..., :init_audio_len] = init_audio_tokens
        for phase, n_steps in zip(range(self.nq), decoding_steps):
            audio_tokens = self._generate_phase(
                audio_tokens=audio_tokens,
                conditioning_tensor=conditioning_tensor,
                phase=phase,
                num_decoding_steps=n_steps,
                device=device,
                init_audio_len=init_audio_len,
                init_audio_tokens=init_audio_tokens,
                top_k=top_k,
                top_p=top_p
            )
        return audio_tokens

    @torch.inference_mode()
    def _generate_phase(
        self,
        audio_tokens:Tensor, # (B, Nq, T)
        conditioning_tensor:Tensor, # (B, N, D=512)
        phase:int,
        num_decoding_steps:int, # `s` in the paper
        device:torch.device,
        temp:float = 3.0,
        anneal_temp:bool = True,
        top_k:tp.Optional[int] = None,
        top_p:tp.Optional[float] = None,
        init_audio_len:int = 0,
        init_audio_tokens:tp.Optional[Tensor] = None
    ) -> Tensor:
        B, Nq, T = audio_tokens.shape
        gen_shape = (B, 1, T)

        phase_gen_tokens = torch.full(gen_shape, self.mask_id, device=device)

        num_chunks = T//self.spanlen
        chunked_shape = (B, 1, num_chunks)
        DONT_REMASK_SCORE = -float("inf")

        if T % self.spanlen != 0:
            # make `T` divisible by `spanlen` and truncate the `audio_tokens`
            T = self.spanlen * num_chunks
            audio_tokens = audio_tokens[..., :T]
            phase_gen_tokens = phase_gen_tokens[..., :T]
        
        num_init_audio_chunks = init_audio_len // self.spanlen
        # `span_scores`: high score if less probablity in the sampled tokens
        span_scores = torch.zeros(chunked_shape, dtype=torch.float32, device=device) # (B, 1, num_chunks)
        # less score assigned to the initial audio chunks; so that they are not masked 
        # (only least probable spans are masked so that they are predicted by the model)
        span_scores[..., :num_init_audio_chunks] = DONT_REMASK_SCORE
        num_chunks_to_generate = num_chunks - num_init_audio_chunks

        for timestep in range(num_decoding_steps): # timestep is `i` from the paper
            mask_p = math.cos((math.pi * timestep)/(2*num_decoding_steps))
            # `num_spanned_mask` decreases as `timestep` increases
            num_spanned_mask = max(1, int(num_chunks_to_generate * mask_p))
            
            # mask the least probable spans
            ## returns topk `span_scores` indices; which correspond to tokens which are least probable
            least_tok_prob_idx = span_scores.topk(k=num_spanned_mask, dim=-1).indices
            chunk_mask = torch.zeros(chunked_shape, dtype=torch.float32, device=device).bool() # False
            ## the topk indices are set to True i.e least probable spans are True, rest are False
            chunk_mask.scatter_(dim=-1, index=least_tok_prob_idx, value=True)
            ## `chunk_mask` is repeated `spanlen` times to get actual mask
            mask = chunk_mask.repeat_interleave(repeats=self.spanlen, dim=-1)
            ## apply `mask` to `phase_gen_tokens`, least probable tokens set to `self.mask_id`
            phase_gen_tokens[mask] = self.mask_id

            if init_audio_tokens is not None:
                phase_gen_tokens[..., :init_audio_len] = init_audio_tokens[:, [phase], :]

            audio_tokens[:, [phase], :] = phase_gen_tokens
            _temp = max(
                temp*(1-timestep/num_decoding_steps) if anneal_temp else temp, 0.01
            )

            logits:Tensor = self(audio_tokens, conditioning_tensor)[:, [phase], :, :] # (B, 1, T, cardinality) <= (B, Nq, T, cardinality)

            tempr_logits = logits/_temp
            probs = tempr_logits.softmax(dim=-1) # (B, 1, T, cardinality)
            # sample tokens from logits where least probable tokens are masked
            generated_tokens = MAGNET._sample_tokens( # (B, T, 1)
                probs[:, 0, :, :], # (B, T, cardinality)
                top_k=top_k, top_p=top_p
            )

            # place generated tokens in place of masked ids, else keep the original tokens
            # `mask`: True where top-k token; False when least probable token
            mask = (phase_gen_tokens == self.mask_id) # (B, 1, T)
            #                  (B, 1, T)          # (B, 1, T)      # (B, T, 1)[..., 0]
            # phase_gen_tokens when condition is True else generated_tokens[..., 0]
            # generated_tokens[..., 0] when mask is True else phase_gen_tokens
            # print(mask.shape, phase_gen_tokens.shape, generated_tokens[..., None, 0].shape)
            # print(audio_tokens[:, [phase], :].shape)
            phase_gen_tokens = torch.where(
                (mask==False)[:, 0], # (B, T)
                phase_gen_tokens[:, 0], # (B, T)
                generated_tokens[..., 0] # (B, T)
            )[:, None] # (B, 1, T)
            # print(phase_gen_tokens.shape)
            audio_tokens[:, [phase], :] = phase_gen_tokens # (B, Nq, T)
            # print(phase_gen_tokens.shape)
            # probs of generated tokens
            #              (B, 1, T, cardinality)      # (B, T, 1).unsqueeze(1) => (B, 1, T, 1)
            sampled_tok_probs = probs.gather(dim=-1, index=generated_tokens.unsqueeze(1))[..., 0] # (B, 1, T) <= (B, 1, T, 1)
            # {1 - (max prob in the span)} => high score if less probablity in the sampled tokens
            # (B, 1, num_chunks) <= (B, 1, num_chunks, T//num_chunks) <= (B, 1, T)
            span_scores = 1 - sampled_tok_probs.reshape((B, 1, num_chunks, -1)).max(-1).values
            # chunk_mask: the least probable spans are True, the rest i.e unmasked positions are False
            # False positions are assigned DONT_REMASK_SCORE so they are not masked in the next iteration
            # (only least probable spans are masked so that they are predicted by the model)
            span_scores.masked_fill_(chunk_mask==False, DONT_REMASK_SCORE)
        return audio_tokens

    @staticmethod
    def _sample_tokens(
        probs:Tensor,
        top_k:tp.Optional[int]=None, 
        top_p:tp.Optional[float]=None
    ) -> Tensor:
        sampler:tp.Callable[[Tensor], Tensor]
        if top_k is not None and top_p is not None:
            raise ValueError(
                "Either `top_k` or `top_p` should be provided, both together are not yet supported."
            )
        elif top_k is not None:
            sampler = lambda probs: sample_tokens.topK(probs, k=top_k)
        elif top_p is not None:
            sampler = lambda probs: sample_tokens.topP(probs, p=top_p)
        else:
            sampler = lambda probs: sample_tokens.multinomial(probs, num_samples=1)
        return sampler(probs)


def get_magnet_model():
    # (B, Nq, T) # (B, N, cond_dim) # (T, T)
    model = Transformer(monfig)
    model = MAGNET(model, monfig) # build model, has lazy modules
    x = torch.randint(0, 1024, (2, 4, 750))
    conditioning_tensor = torch.randn((2, 10, 512))
    ___ = model(x, conditioning_tensor)
    return model

from icecream import ic
ic.configureOutput(outputFunction=lambda x: print(x, end="\n\n"))

# class TransformerDecoder(nn.Module):
#     """```
#     Args: config
#     Input:
#         seq:Tensor,               # (B, T, d_model)  # q
#         cond_tensor:Tensor,       # (B, N, cond_dim) # kv # output from t5-encoder model
#         cond_padding_mask:Tensor, # (B, N) # prompt padding mask
#         cross_att_mask:Tensor     # (T, N) # mask in cross attention due to padding in cond_tensor
#     ```"""
#     def __init__(self, config:monfig):
#         super().__init__()
#         self.transformer_decoder = nn.TransformerDecoder(
#             decoder_layer=nn.TransformerDecoderLayer(
#                 d_model=config.d_model,
#                 nhead=config.num_heads,
#                 dim_feedforward=config.d_model*4,
#                 dropout=config.dropout_rate,
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=config.num_layers,
#             norm=nn.LayerNorm(config.d_model)
#         )

#     def forward(
#         self,
#         seq:Tensor,
#         cond_tensor:tp.Optional[Tensor], # Optionally None in case of cfg
#         attn_mask:Tensor,
#         cond_padding_mask:tp.Optional[Tensor] = None,
#         cross_att_mask:tp.Optional[Tensor] = None
#     ) -> Tensor:
#         return self.transformer_decoder(
#             tgt=seq, # sequence to decoder
#             memory=cond_tensor, # sequence from last layer of encoder # cross att shape (T, N)
#             tgt_mask=attn_mask,
#             memory_key_padding_mask=cond_padding_mask, # ??? prompt padding mask, from T5 model
#             memory_mask=cross_att_mask # ??? mask in cross attention due to padding in cond_tensor
#         ) # (B, T, d_model)