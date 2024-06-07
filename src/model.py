import dataclasses as dc
import inspect
import typing as tp

import torch
from torch import (
    nn, Tensor
)

@dc.dataclass
class monfig: # model config
    d_model:int
    num_heads:int
    dropout_rate:float

    cardinality:int
    num_codebooks:int

    max_seq_len:int


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
            norm=nn.LayerNorm(config.d_model)
        )

    def forward(
        self,
        seq:Tensor,
        cond_tensor:Tensor,
        cond_padding_mask:Tensor,
        cross_att_mask:Tensor
    ) -> Tensor:
        return self.transformer_decoder(
            tgt=seq, # sequence to decoder
            memory=cond_tensor, # sequence from last layer of encoder # cross att shape (T, N)
            memory_key_padding_mask=cond_padding_mask, # prompt padding mask, from T5 model
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
        self.nq = config.num_codebooks # nq

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

    # conditioning_tensor: (B, N, cond_dim)
    def forward(
        self,
        x:Tensor, # (B, Nq, T)
        conditioning_tensor:Tensor, # (B, N, cond_dim)
        seq_mask:tp.Optional[Tensor]=None, # (B, T)
        cross_att_mask:tp.Optional[Tensor]=None # (T, N)
    ):
        # emb: (cardinality {in dimension T}, d_model) ==x[:, codebook]: (B, T)=> (B, T, d_model)
        x = sum([self.embeddings[codebook](x[:, codebook]) for codebook in range(self.nq)]) # (B, T, d_model)
        x = self.model(x, conditioning_tensor, seq_mask, cross_att_mask) # (B, T, d_model)
        # stack((B, T, cardinality), dim=1) => (B, nq, T, cardinality)
        x = torch.stack([self.linears[codebook](x) for codebook in range(self.nq)], dim=1)
        return x
    
    @torch.no_grad()
    def generate(
        self, 
        prompt:tp.Optional[str]=None,
    ):
        # generate in eval mode
        self.eval()
        pass

    def _generate_stage(self):
        pass

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
