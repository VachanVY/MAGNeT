import typing as tp
import math
import bisect
import random as r
import dataclasses as dc
import contextlib as ctxlib
import os
import time

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchaudio

from encodec import (
    EncodecModel,
    utils as enc_utils
)
from transformers import (
    AutoTokenizer, T5EncoderModel
)

from music_bench import (
    shuffle_preserve_order,
    ioPathTextDs,
    AUDIO_TXT_PATH
)
from model import (
    monfig, 
    MAGNET, 
    TransformerDecoder
)
from lr_scheduler import CosineDecayWithWarmup

#---------------------------------------------------------------------------------------------------#
@dc.dataclass
class tonfig:
    # general train args
    num_steps:int
    batch_size:int
    num_grad_accumalation_steps:int
    log_interval:int

    # Early stopping args
    patience:int
    restore_best_weights:bool

    # MAGNeT and Model args
    spanlen:int
    seqlen:int = monfig.max_seq_len
    attn_window:int

    # vocab (without mask_id): {0, ..., 1023} => mask_id = 1024; cardinality = len(vocab) which is 1024
    mask_id:int = monfig.cardinality

    # cosine decay with warmup args
    warmup_steps:int
    max_learning_rate:float
    decay_steps:int
    min_learning_rate:float

    # optimizer args
    decay:bool = True # if true uses cosine decay else max_learning_rate thoughtout training
    weight_decay:float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    clipnorm:float = 1e0
    ## ema_momentum:float = 0.99

    # eval and checkpointing args
    eval_freq:int
    eval_steps:int
    init_from:str = 0
    ckpt_dir:str
    always_checkpoint:bool = False

    # device args
    device:str = "cuda" # 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
    device_type:str = "cuda" if "cuda" in device else "cpu"

    # dtype args
    dtype:str = "float16"


encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(3.0) # 3 kbps (Nq = 4)
SRATE = encodec_model.sample_rate
MAX_SEC = 10
WAVLEN = int(SRATE*MAX_SEC)
QCODING_LEN = 750
print("compiling encodec model...")
encodec_model.to("cuda")
encodec_model = torch.compile(encodec_model)
print("compiled encodec model.")
encodec_model.eval() # in eval mode


## (google-t5/t5-small) (google-t5/t5-base) (google-t5/t5-large) (google-t5/t5-3b) (google-t5/t5-11b)
T5_MODEL_PATH = "google-t5/t5-small"
cond_model = T5EncoderModel.from_pretrained(T5_MODEL_PATH)
print("compiling T5 model...")
cond_model.to("cuda")
cond_model = torch.compile(cond_model)
print("compiled T5 model.")
cond_model.eval() # in eval mode

cond_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
#---------------------------------------------------------------------------------------------------#

class PreProOps:
    @staticmethod
    def preprocess_wavpath(wavpath:str, squeeze:bool=True):
        wav, sr = torchaudio.load(wavpath)
        wav = enc_utils.convert_audio(
                wav, sr=sr, target_sr=SRATE,
                target_channels=encodec_model.channels
            )
        if squeeze:
            return wav.squeeze()
        return wav
    
    @staticmethod
    def getQuantizedCodings(bs_waves:torch.Tensor): # (B, C=1, T=WAVLEN)
        with torch.no_grad():
            encoded_frames = encodec_model.encode(bs_waves)
        codes = encoded_frames[0][0] # (B, n_q=4, T=750)
        return codes
    
    @staticmethod
    def getAudioFromCodings(codes:torch.Tensor):
        encoded_frames = [(codes, None)]
        with torch.no_grad():
            wavs = encodec_model.decode(encoded_frames)
        return wavs
    
    @staticmethod
    def get_conditioned_tensor(padded_cond_seq:list[str]) -> torch.Tensor:
        with torch.no_grad():
            cond_tensors = cond_model(
                input_ids=padded_cond_seq["input_ids"], # (B, N)
                attention_mask=padded_cond_seq["attention_mask"] # (B, N)
            ) # (B, N, cond_dim)
        return cond_tensors


class PreProDataset:
    def __init__(self, wav_paths:list[str], texts:list[str], audio_pad_id:int, wavlen:int):
        self.wav_paths = wav_paths # (N//B, B)
        self.texts = texts # (N//B, B)
        self.audio_pad_id = audio_pad_id
        self.wavlen = wavlen
    
    def preprocess(self, wavpaths:str, text_str:str):
        # pads to maxlen in batch
        bs_wavs = list(map(PreProOps.preprocess_wavpath, wavpaths))
        bs_wavs = nn.utils.rnn.pad_sequence(
            sequences=bs_wavs,
            batch_first=True,
            padding_value=self.audio_pad_id
        ) # (B, 1, len) where len is the maximim length in the list of wavs
        # quatized codings
        quantized_codings = PreProDataset.getQuantizedCodings(
            bs_waves=bs_wavs.unsqueeze(1) # add channel dimension
        ) # (B, Nq=4, <=750)
        
        # if maxlen<WAVLEN pad to WAVLEN
        if quantized_codings.shape[-1] != self.wavlen: # if not 750
            quantized_codings = F.pad(
                quantized_codings, (0, self.wavlen - bs_wavs.shape[-1]), value=self.audio_pad_id
            ) # (B, Nq=4, 750)
        pad_mask = quantized_codings != self.audio_pad_id  # (B, T) # False value contains the padded tokens # True value is to be taken
        # Example = [[0,    1,   2, 1024, 1024],
        #            [1023, 2, 100,    4, 1024]]
        # example < 1024 =>
        # [[True, True, True, False, False],
        #  [True, True, True, True, False]]
        
        text_toks = cond_tokenizer(
            text_str, return_tensors="pt", padding=True
        )
        return {"wav": bs_wavs, "mask": pad_mask}, text_toks
    
    def iter_batches(self):
        while True:
            self.wav_paths, self.texts = shuffle_preserve_order(self.wav_paths, self.texts)
            for batched_wavpath, batched_text_str in zip(self.wav_paths, self.texts):
                wavs, text_toks = self.preprocess(batched_wavpath, batched_text_str)
                yield wavs, text_toks


class MagnetTrainer:
    def __init__(self, config:tonfig):
        self.magnet_model = magnet_model
        self.span_len = config.spanlen
        self.mask_id = config.mask_id
    
        self._num_spans_to_mask = torch.tensor(
            self._get_number_of_spans_to_mask(T=config.seqlen, L=config.spanlen)
        )
        self._att_mask = self._magnet_restricted_att_mask(
            shape=(config.seqlen, config.seqlen),
            windows=(config.attn_window, config.attn_window)
        )

    def _get_number_of_spans_to_mask(self, T:int, L:int):
        """
        ### Docs for how to use output
        ```
        We need something sort of like this {0: 0, 0.01: 3, ..., 0.99: 588, 1.0: 748} (dummy_mask_rate:u) but
        [0, 3, ..., 588, 748] <= num_spans_to_mask (output)
        [0, 1, ..., 99 , 100] <= corresponding indexes of num_spans_to_mask (mask_rate*100)
        so no need of a dictionary like above
        ```"""
        approx_maskrate_given_u = lambda u: 1 - math.comb(T-L, u)/math.comb(T, u) # LHS
        approx_mask_rates = [approx_maskrate_given_u(u) for u in range(0, T)]
        
        # get mean number of spans to mask (u) given maskrate ([i/100 for i in range(100+1)])
        num_spans_to_mask = []
        for mask_percentage in range(100+1):
            mask_rate = mask_percentage/100 # dummy mask rates
            # Return the index where to insert mask_rate in approx_mask_rates (sorted) to keep it sorted
            u = bisect.bisect_left(approx_mask_rates, mask_rate)
            num_spans_to_mask.append(u)
        return num_spans_to_mask # (100,)

    def _get_spanned_mask(self, mask_rates:Tensor, seq_len:int):
        """```
        True value will be masked
        Args:
            mask_rates: Tensor => mask_probabilities of shape (B,)
            T: int => length of sequence
        Returns:
            mask: Tensor => mask of shape (B, T)
        ```"""
        B = len(mask_rates)

        indexes = torch.round(mask_rates*100) # indexes to take from num_spans_to_mask # (B,)
        # contains number of spans to mask
        batched_num_spans = torch.tensor(
            self._num_spans_to_mask
        ).gather(dim=0, index=torch.tensor(indexes)).clip(min=1) 
        
        batch_randperm = torch.rand((B, seq_len)).argsort(-1) # rand integers from 0 to T
        mask = batch_randperm < batched_num_spans[..., None] # contains batched_num_spans number of Trues
        shifted_mask = mask.clone()
        for _ in range(self.span_len-1):
            shifted_mask = torch.concat((torch.full((B, 1), False), shifted_mask[:, :-1]), dim=-1)
            mask = torch.logical_or(mask, shifted_mask)
        return mask # (B, T) # True value will be masked
    
    # from xformers.ops.fmha.attn_bias.LocalAttentionFromBottomRightMask
    def _magnet_restricted_att_mask(self, shape:tuple, windows:tuple, dtype:torch.dtype=torch.float32):
        create_as = dtype if dtype is not torch.bfloat16 else torch.float32
        mask = torch.full(
            shape, dtype=create_as, fill_value=1,
        )

        num_queries, num_keys = shape[-2:]
        shift = num_keys - num_queries

        mask = torch.triu(mask, diagonal=shift - windows[0])
        mask = torch.tril(mask, diagonal=shift + windows[1])
        mask = torch.log(mask)
        return mask.to(dtype)

    # TODO: Implement this
    def _cfg(self):
        raise NotImplementedError

    def _magnet_cross_entropy(
        self,
        y_true:Tensor,    # (B, nq, T)
        y_pred:Tensor,    # (B, nq, T, cardinality) # logits
        loss_mask:Tensor, # (B, nq, T) # take loss only on masked tokens (True value)
        phase:int
    ):
        y_true, loss_mask = y_true[:, phase], loss_mask[:, phase] # (B, T)
        y_pred = y_pred[:, phase] # (B, T, cardinality)

        # take loss only on masked tokens (True value) # so mask the False value with -1
        y_true[loss_mask==False] = -1
        loss = F.cross_entropy(
            y_pred, y_true, ignore_index=-1
        )
        return loss

    def mini_train_step(
        self,
        audio_input:dict[str, Tensor],
        padded_cond_seq:dict[str, Tensor],
    ):
        """```
        audio_input: {"wav": bs_wavs,   # tensor of shape (B, T)
                      "mask": pad_mask} # tensor of shape (B, T)
        conditioning_text: batch(text)
        ```"""
        # audio tensor and mask
        (
            audio_tokens,
            audio_pad_mask # False value contains the padded tokens # True value is to be taken # loss taken on True values
        ) = audio_input["wav"], audio_input["mask"]

        # Batch_dim, Codebooks, Time_dim
        B, Nq, T = audio_tokens.shape

        # conditioning tensor
        cond_tensor = PreProOps.get_conditioned_tensor(padded_cond_seq)

        # A random phase: for masking a random codebook
        phase = r.randint(0, Nq-1)
        
        # masking rate = gamma(i, s) = cos(pi*(i-1)/(2*s)) where i is the decoding time step i ϵ [1, s]
        # during training time step i is randomly sampled, thus (i-1)/s is randomly sampled
        rand_decoding_time = torch.rand(B) # (B,)
        mask_rates = torch.cos(math.pi*rand_decoding_time/2) # (B,)

        # mask the tokens of the current phase in codebook # True value will be masked
        phase_mask = self._get_spanned_mask(mask_rates, seq_len=T) # (B, T)
        idx0 = torch.arange(B)
        idx1 = torch.full_like(idx1, phase)
        
        # add phase mask to final_mask
        final_mask = torch.zeros_like(audio_tokens).bool().index_put(
            indices=[idx0, idx1], # [dim0 idx, dim1 idx]
            values=phase_mask
        )
        
        # mask for loss
        loss_mask = final_mask.clone()
        # mask padded values in loss mask # take loss only on masked tokens (True value)
        # don't take loss on padded tokens
        loss_mask &= audio_pad_mask

        # mask all codebooks greater than phase
        final_mask[:, (phase+1):, :] = torch.zeros((B, Nq-(phase+1), T)).bool() # (B, Nq, T)
        
        # use final_mask on audio tokens
        audio_tokens = torch.where(final_mask, self.mask_id, audio_tokens) # (B, Nq, T)
        
        logits:Tensor = self.magnet_model(
            x=audio_tokens,
            conditioning_tensor=cond_tensor,
            seq_mask=None,
            cross_att_mask=None
        ) # (B, Nq, T, cardinality)

        accuracy = torch.sum( # (B, Nq, T) => (B,)
            logits.argmax(-1) == audio_tokens, dim=(-1, -2)
        ).mean() # (B,) => ()
        loss = self._magnet_cross_entropy(
            y_true=audio_tokens, y_pred=logits,
            loss_mask=loss_mask, phase=phase
        )
        return loss, accuracy
    

# dataset
dataset = ioPathTextDs(
    save_path=AUDIO_TXT_PATH,
    batch_size=tonfig.batch_size,
    split_float=0.9
)
X_train, y_train = dataset["train"]
X_val, y_val = dataset["val"]

train_iterator, val_iterator = (
    PreProDataset(X_train, y_train, audio_pad_id=tonfig.mask_id).iter_batches(),
    PreProDataset(X_val, y_val).iter_batches()
)
# lr scheduler
get_lr = CosineDecayWithWarmup(
    warmup_steps=tonfig.warmup_steps,
    max_learning_rate=tonfig.max_learning_rate,
    decay_steps=tonfig.decay_steps,
    min_learning_rate=tonfig.min_learning_rate
)
# for float16 or bfloat16 training
ctx = (
    ctxlib.nullcontext() if "cpu" in tonfig.device_type
    else torch.autocast(
            device_type=tonfig.device_type,
            dtype={"float32"  : torch.float32,
                    "bfloat16": torch.bfloat16,
                    "float16" : torch.float16}[tonfig.dtype]
        )
)

os.makedirs(tonfig.ckpt_dir, exist_ok=True)
if "scratch" in tonfig.init_from:
    model_config = monfig
    model_args = dc.asdict(monfig)
    magnet_model = MAGNET(
        model=TransformerDecoder(model_config),
        config=model_config
    )

    train_from_step = 0
    best_val_loss = 1e9
    checkpoint = None
elif "resume" in tonfig.init_from:
    ckpt_path = os.path.join(tonfig.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=tonfig.device)

    model_args = checkpoint["model_args"]
    model_config = monfig(**model_args)
    magnet_model = MAGNET(
        model=TransformerDecoder(model_config),
        config=model_config
    )
    state_dict:dict[str, Tensor] = checkpoint["model_state"]
    # TODO/NOTE: Is this needed?
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    magnet_model.load_state_dict(state_dict)
    train_from_step = checkpoint["step"]
    best_val_loss = checkpoint["best_val_loss"]

magnet_model.to(tonfig.device)

scaler = torch.cuda.amp.GradScaler(enabled=(tonfig.device=="float16"))
optimizer = magnet_model.configure_optimizers(
    weight_decay=tonfig.weight_decay,
    learning_rate=tonfig.max_learning_rate,
    betas=(tonfig.beta1, tonfig.beta2),
    device_type=tonfig.device_type
)
if ("resume" in tonfig.init_from) and ("optimizer" in checkpoint):
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None # free memory

magnet_trainer = MagnetTrainer(
    magnet_model=magnet_model,
    cond_model=cond_model,
    config=tonfig
)


@torch.no_grad()
def evaluate():
    magnet_model.eval()

    mean_losses, mean_metrics = [], []
    for eval_iterator in [train_iterator, val_iterator]:
        losses = torch.empty((tonfig.eval_steps,))
        metrics = torch.empty_like(losses)

        for eval_step in range(tonfig.eval_steps):
            audio_input, cond_text = next(eval_iterator)
            with ctx:
                loss, acc = magnet_trainer.mini_train_step(
                    audio_input=audio_input, 
                    padded_cond_seq=cond_text
                )
            losses[eval_step], metrics[eval_step] = loss, acc
        mean_losses.append(losses.mean())
        mean_metrics.append(metrics.mean())

    magnet_model.train()
    return mean_losses, mean_metrics


def train():
    audio_input, cond_text = next(train_iterator)

    print("Training about to start...")
    t0 = time.time()
    wait = 0; best_step = 0 # for early stopping
    for step in range(0, tonfig.num_steps):
        # evaluate 
        if step % tonfig.eval_freq == 0 and step > 0:
            mean_losses, mean_accuracies = evaluate()
            print(
                    f"\t| Training Loss: {mean_losses[0]:.4f} || Training Accuracy: {mean_accuracies[0]:.4f} |" 
                    f"| Validation Loss: {mean_losses[1]:.4f} || Validation Accuracy: {mean_accuracies[1]:.4f} |"
                )
            if mean_losses[1] < best_val_loss or tonfig.always_checkpoint:
                best_val_loss = mean_losses[1]
                wait = 0; best_step = step

                checkpoint = {
                    "model_state": magnet_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_args": model_args,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "train_config": dc.asdict(tonfig)
                }
                print(f"saving checkpoint to {tonfig.ckpt_dir}")
                torch.save(checkpoint, os.path.join(tonfig.ckpt_dir, "ckpt.pt"))
            else:
                wait += 1

        # set learning rate for all params
        lr = get_lr(step) if tonfig.decay else tonfig.max_learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # gradient accumulation step
        for mini_step in range(tonfig.num_grad_accumalation_steps):
            with ctx:
                loss, accuracy = magnet_trainer.mini_train_step(
                    audio_input=audio_input, conditioning_text=cond_text
                )/tonfig.num_grad_accumalation_steps
                # async prefetch immediately
                audio_input, cond_text = next(train_iterator)

            # keeps on scaling and adding gradients when calling backward
            scaler.scale(loss).backward()

        if tonfig.clipnorm is not None:
            # unscale the gradients
            scaler.unscale_(optimizer)
            # clips gradients in-place to grad norm
            nn.utils.clip_grad_norm_(magnet_model.parameters(), max_norm=tonfig.clipnorm)

        # calls unscale to the optimizer unless already called, checks for infs and nans as a part of unscale_
        # calls optimizer.step on unscaled grads if no infs and nans else optimizer.step is skipped
        scaler.step(optimizer)
        # Update the scale factor
        scaler.update()

        # flush grads to save memory
        optimizer.zero_grad(set_to_none=True)

        # some logging
        t1 = time.time()
        dt = t1-t0
        t0 = t1
        if step % tonfig.log_interval:
            # multiply as loss was scaled for gradient accumulation
            lossf = loss.item() * tonfig.num_grad_accumalation_steps
            print(
                f"| Step: {step} || Loss: {lossf:.4f} || Accuracy: {accuracy:.4f} |"
                f"| LR: {lr:e} || dt: {dt*1000:.2f}ms |"
            )

        if wait > tonfig.patience:
            print(
                f"Early Stopping at step: {step}."
                f"Best Validation Loss: {best_val_loss}"
                f"Best step: {best_step}"
            )
            if tonfig.restore_best_weights:
                model_state = torch.load(
                    f=os.path.join(tonfig.ckpt_dir, "ckpt.pt"),
                    map_location=tonfig.device
                )["model_state"]
                magnet_model.load_state_dict(model_state)
            break

if __name__ == "__main__":
    train()
