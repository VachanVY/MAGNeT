import dataclasses as dc
import inspect
import typing as tp
import math

import torch
from torch import (
    nn, Tensor
)

from music_bench import QCODING_LEN
import utils.sample_tokens as sample_tokens

@dc.dataclass
class monfig: # model config
    d_model:int = 512
    num_heads:int = 8
    dropout_rate:float = 0.0
    num_layers:int = 8

    cardinality:int = 1024 # sorta like vocab_size
    num_codebooks:int = 4

    max_seq_len:int = QCODING_LEN


class TransformerDecoder(nn.Module):
    """```
    Args: config
    Input:
        seq:Tensor,               # (B, T, d_model)  # q
        cond_tensor:Tensor,       # (B, N, cond_dim) # kv # output from t5-encoder model
        cond_padding_mask:Tensor, # (B, N) # prompt padding mask
        cross_att_mask:Tensor     # (T, N) # mask in cross attention due to padding in cond_tensor
    ```"""
    def __init__(self, config:monfig):
        super().__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model*4,
                dropout=config.dropout_rate,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model)
        )

    def forward(
        self,
        seq:Tensor,
        cond_tensor:tp.Optional[Tensor], # Optionally None in case of cfg
        cond_padding_mask:tp.Optional[Tensor] = None,
        cross_att_mask:tp.Optional[Tensor] = None
    ) -> Tensor:
        return self.transformer_decoder(
            tgt=seq, # sequence to decoder
            memory=cond_tensor, # sequence from last layer of encoder # cross att shape (T, N)
            memory_key_padding_mask=cond_padding_mask, # ??? prompt padding mask, from T5 model
            memory_mask=cross_att_mask # ??? mask in cross attention due to padding in cond_tensor
        ) # (B, T, d_model)


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
        model:nn.Module,
        config:monfig # Model config
    ):
        self.maxlen = config.max_seq_len
        self.nq = config.num_codebooks # nq
        self.cardinality = config.cardinality

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
        seq_mask:tp.Optional[Tensor]=None, # (B, T)
        cross_att_mask:tp.Optional[Tensor]=None # (T, N)
    ):
        # emb: (cardinality {in dimension T}, d_model) ==x[:, codebook]: (B, T)=> (B, T, d_model)
        x = sum([self.embeddings[codebook](x[:, codebook]) for codebook in range(self.nq)]) # (B, T, d_model)
        x = self.model(x, conditioning_tensor, seq_mask, cross_att_mask) # (B, T, d_model)
        # stack((B, T, cardinality), dim=1) => (B, nq, T, cardinality)
        x = torch.stack([self.linears[codebook](x) for codebook in range(self.nq)], dim=1)
        return x
    
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
    
    @staticmethod
    @property
    def spanlen():
        return 3
    
    @torch.inference_mode()
    def generate(
        self,
        prompt:str|list[str],
        tokenizer:tp.Callable[[str], Tensor],
        cond_model:nn.Module,
        device:torch.device,
        init_audio_tokens:tp.Optional[Tensor]=None,
        top_k:tp.Optional[int]=None,
        top_p:tp.Optional[float]=None,
        decoding_steps:list[int] = [20, 10, 10, 10]
    ) -> Tensor:
        # generate in eval mode
        assert not self.training, "generation should be in eval mode"
        assert prompt is not None, "prompt shouldn't be None... until CFG is implemented"
        if init_audio_tokens is not None and isinstance(init_audio_tokens, Tensor):
            init_audio_tokens = init_audio_tokens.to(device)
            init_audio_len = init_audio_tokens.shape[-1]
            assert init_audio_tokens.shape[-2] == self.nq and init_audio_len < self.maxlen

        tok_prompt:dict[str, Tensor] = tokenizer(
            [prompt] if isinstance(prompt, str) else prompt, 
            padding=True, return_tensors="pt"
        ) # (B, N)
        B = tok_prompt["input_ids"].shape[0]
        cond_model.to(device)
        conditioning_tensor = cond_model(
            input_ids=tok_prompt["input_ids"].to(device),
            attention_mask=tok_prompt["attention_mask"].to(device)
        ) # (B, N, D=512)
        
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
            phase_gen_tokens = phase_gen_tokens.where(mask==False, generated_tokens[..., 0]) # (B, 1, T)
            audio_tokens[:, [phase], :] = phase_gen_tokens # (B, Nq, T)

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
            sampler = lambda logits: sample_tokens.topK(logits=logits, k=top_k)
        elif top_p is not None:
            sampler = lambda logits: sample_tokens.topP(logits=logits, p=top_p)
        else:
            sampler = lambda logits: sample_tokens.multinomial(logits.softmax(dim=-1), num_samples=1)
        return sampler(probs)
    
