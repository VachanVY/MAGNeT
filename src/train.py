import typing as tp
import math
import bisect
import random as r
import dataclasses as dc
import os
import time
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .music_bench import (
    shuffle_preserve_order,
    ioPathTextDs,
    AUDIO_TXT_PATH,
    QCODING_LEN,
    MAX_SEC
)
from .preprocess_ops import PreProOps
from .model import (
    monfig,
    MAGNET,
    Transformer
)
from .utils.lr_scheduler import CosineDecayWithWarmup


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#---------------------------------------------------------------------------------------------------#
@dc.dataclass
class tonfig:
    # general train args
    num_steps:int = 10000 # 1_000_000
    batch_size:int = 64
    num_grad_accumalation_steps:int = 1
    log_interval:int = 1

    # Early stopping args
    patience:int = 10
    restore_best_weights:bool = True

    # MAGNeT and Model args
    spanlen:int = 3 # 3
    ## equal to the maximum length of the quantized codings for 10 seconds of audio
    seqlen:int = QCODING_LEN # 750
    attn_window:int = 5

    # vocab (without mask_id): {0, ..., 1023} => mask_id = 1024
    # cardinality = len(vocab) which is 1024
    mask_id:int = monfig.cardinality

    # cosine decay with warmup args
    warmup_steps:int = 1250
    max_learning_rate:float = ...
    decay_steps:int = num_steps
    min_learning_rate:float = ...

    # optimizer args
    decay:bool = True # if true uses cosine decay else max_learning_rate thoughtout training
    weight_decay:float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    clipnorm:float = 1e0
    ## ema_momentum:float = 0.99

    # eval and checkpointing args
    eval_freq:int = 2000
    eval_steps:int = 200
    init_from:str = 0
    ckpt_dir:str = "ckpts"
    always_checkpoint:bool = False

    # device args
    device:str = "cuda" # 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
    device_type:str = "cuda" if "cuda" in device else "cpu"

    # dtype args # 'encodec model has LSTMs which doesn't work with bfloat16'
    dtype:str = "float16" # float16, float32
#---------------------------------------------------------------------------------------------------#


class PreProDataset:
    def __init__(
        self, 
        wav_paths:list[str], 
        texts:list[str], 
        audio_pad_id:int, 
        qcoding_len:int,
        preprocess_ops:PreProOps,
        device:str
    ):
        self.wav_paths = wav_paths # (N//B, B)
        self.texts = texts # (N//B, B)
        self.audio_pad_id = audio_pad_id
        self.qcoding_len = qcoding_len
        self.device = torch.device(device)
        self.preprocess_ops = preprocess_ops
    
    def preprocess(self, wavpaths:str, text_str:str):
        # pads to maxlen in batch
        quantized_codings = list(map(
            lambda x: self.preprocess_ops.preprocess_wavpath(x, get_qcodings=True), 
            wavpaths)
        )# [[T=..., Nq=4], ...]

        quantized_codings = nn.utils.rnn.pad_sequence(
            sequences=quantized_codings,
            batch_first=True,
            padding_value=self.audio_pad_id
        ).movedim(1, -1) # (B, Nq=4, T<=750) <= (B, T<=750, Nq=4)
        
        # if maxlen<WAVLEN pad to WAVLEN
        if quantized_codings.shape[-1] != self.qcoding_len: # if not 750
            quantized_codings = F.pad(
                quantized_codings, (0, self.qcoding_len - quantized_codings.shape[-1]), value=self.audio_pad_id
            ) # (B, Nq=4, 750)
        pad_mask = quantized_codings != self.audio_pad_id  # (B, T) # False value contains the padded tokens # True value is to be taken
        # Example = [[0,    1,   2, 1024, 1024],
        #            [1023, 2, 100,    4, 1024]]
        # example != 1024 =>
        # [[True, True, True, False, False],
        #  [True, True, True, True, False]]
        
        return {"qcode": quantized_codings, "mask": pad_mask}, text_str
    
    def iter_batches(self):
        while True:
            self.wav_paths, self.texts = shuffle_preserve_order(self.wav_paths, self.texts)
            for batched_wavpath, batched_text_str in zip(self.wav_paths, self.texts):
                wavs, text_str = self.preprocess(batched_wavpath, batched_text_str)
                yield wavs, text_str

class MagnetTrainer:
    def __init__(
        self, 
        magnet_model:MAGNET, 
        preprocess_ops:PreProOps, 
        config:tonfig,
    ):
        self.magnet_model = magnet_model
        self.span_len = config.spanlen
        self.mask_id = config.mask_id
        self.device = torch.device(config.device)
    
        self._num_spans_to_mask = torch.tensor(
            self._get_number_of_spans_to_mask(T=config.seqlen, L=config.spanlen),
            device=self.device
        ) # (100,)
        self.preprocess_ops = preprocess_ops

    def _get_number_of_spans_to_mask(self, T:int, L:int):
        """
        ### Docs for how to use output
        We need something sort of like this {0: 0, 0.01: 3, ..., 0.99: 588, 1.0: 748} (dummy_mask_rate:u) but\n
        [0, 3, ..., 588, 748] <= num_spans_to_mask (output)\n
        [0, 1, ..., 99 , 100] <= corresponding indexes of num_spans_to_mask (mask_rate*100)\n
        so no need of a dictionary like above
        """
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
        """
        Returns a mask of shape (B, T) where the True value will be masked.\n
        Uses `mask_rates` and `self._num_spans_to_mask`\n
        Args:
            `mask_rates`: Tensor => mask_probabilities of shape (B,)
            `T`: int => length of sequence
        Returns:
            `mask`: Tensor => mask of shape (B, T)
        """
        B = len(mask_rates)

        indexes = torch.round(mask_rates*100).to(self.device).to(torch.int64) # indexes to take from num_spans_to_mask # (B,)
        # contains number of spans to mask
        batched_num_spans = self._num_spans_to_mask.gather(dim=0, index=indexes).clip(min=1)
        
        batch_randperm = torch.rand((B, seq_len), device=self.device).argsort(-1) # rand integers from 0 to T
        mask = batch_randperm < batched_num_spans[..., None] # contains `batched_num_spans` number of Trues
        shifted_mask = mask.clone()
        for _ in range(self.span_len-1):
            shifted_mask = torch.concat((torch.zeros((B, 1), device=self.device).bool(), shifted_mask[:, :-1]), dim=-1)
            mask = torch.logical_or(mask, shifted_mask)
        return mask # (B, T) # True value will be masked

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

        # take loss only on masked tokens (True value) # so -1 the False values
        y_true[loss_mask==False] = -1
        """
        print("NUM -1", (y_true==-1).sum(-1))
        print("y_true", y_true.shape, "\n\ny_pred\n", y_pred.shape, "\n\nloss_mask\n", loss_mask.shape, "\n\n")
        print("y_pred.movedim(-1, -2)\n", y_pred.movedim(-1, -2).shape, "\n\ny_true\n", y_true, "\n\n")
        """
        loss = F.cross_entropy(
            y_pred.movedim(-1, -2), # (B, num_classes=cardinatlity, d1=T) <= (B, T, cardinality)
            y_true, # (B, d1=T)
            ignore_index=-1
        )
        return loss

    def mini_train_step( # mini train step if gradient accumulation is used
        self,
        audio_input:dict[str, Tensor],
        padded_cond_str:list[str],
    ):
        """```
        audio_input: {"qcode": audio_tokens,   # tensor of shape (B, Nq, T)
                      "mask": audio_pad_mask} # tensor of shape (B, Nq, T)
        padded_cond_str: batch(text)
        ```"""
        # audio tensor and mask
        (
            audio_tokens,
            audio_pad_mask # False value contains the padded tokens # True value is to be taken # loss taken on True values
        ) = audio_input["qcode"], audio_input["mask"]

        # Batch_dim, Codebooks, Time_dim
        B, Nq, T = audio_tokens.shape

        # conditioning tensor
        padded_cond_seq = self.preprocess_ops.tokenize(padded_cond_str)
        cond_tensor = self.preprocess_ops.get_conditioned_tensor(padded_cond_seq)

        # A random phase: for masking a random codebook
        phase = r.randint(0, Nq-1)
        
        # masking rate = gamma(i, s) = cos(pi*(i-1)/(2*s)) where i is the decoding time step i ϵ [1, s]
        # during training time step i is randomly sampled, thus (i-1)/s is randomly sampled which is `rand_decoding_time`
        rand_decoding_time = torch.rand(B) # (B,)
        mask_rates = torch.cos(math.pi*rand_decoding_time/2) # (B,)

        # mask the tokens of the current phase in codebook # True value will be masked
        phase_mask = self._get_spanned_mask(mask_rates, seq_len=T) # (B, T)
        
        # add phase mask to final_mask
        ## filed with False values only
        final_mask = torch.zeros_like(audio_tokens, device=self.device).bool() # (B, Nq, T)
        ## at index `phase`, fill with phase_mask (True values which will be masked)
        final_mask[:, phase, :] = phase_mask # (B, T)
        
        # mask for loss
        loss_mask = final_mask.clone()
        # mask padded values in loss mask and don't take loss on padded tokens
        # take loss only on masked tokens (True value)
        # audio_pad_mask: False value contains the padded tokens # True value is to be taken
        # see docs/magnet_code.md for explanation on doing the & operation
        loss_mask &= audio_pad_mask # (B, Nq, T)

        # mask all codebooks greater than phase with True values; True values will be masked
        final_mask[:, (phase+1):, :] = torch.ones((B, Nq-(phase+1), T), device=self.device).bool() # (B, Nq, T)
        
        # use final_mask on audio tokens
        # where final_mask is True, replace with mask_id else audio_tokens
        model_input_audio_tokens = torch.where(final_mask, self.mask_id, audio_tokens) # (B, Nq, T)
        
        logits:Tensor = self.magnet_model(
            x=model_input_audio_tokens,
            conditioning_tensor=cond_tensor
        ) # (B, Nq, T, cardinality)

        loss = self._magnet_cross_entropy(
            y_true=audio_tokens, y_pred=logits,
            loss_mask=loss_mask, phase=phase
        )
        # (B, Nq, T) =sum(-1, -2)> (B,) =mean> ()
        """print("SHAPE", logits.shape, audio_tokens.shape)
        print("acc SHAPE", logits.argmax(-1).shape, audio_tokens.shape, 
              "\n\nDTYPE", logits.argmax(-1).dtype, audio_tokens.dtype)
        """
        with torch.no_grad():
            accuracy = ((logits.argmax(-1) == audio_tokens).sum((-1, -2))/(Nq*T)).mean()
        return loss, accuracy


@torch.no_grad()
def evaluate():
    magnet_model.eval()

    mean_losses, mean_metrics = [], []
    for eval_iterator in [train_iterator, val_iterator]:
        losses = torch.empty((tonfig.eval_steps,), device=tonfig.device)
        metrics = torch.empty_like(losses, device=tonfig.device)

        for eval_step in range(tonfig.eval_steps):
            audio_input, cond_text = next(eval_iterator)
            with ctx:
                loss, acc = magnet_trainer.mini_train_step(
                    audio_input=audio_input,
                    padded_cond_str=cond_text
                )
            losses[eval_step], metrics[eval_step] = loss, acc
        mean_losses.append(losses.mean())
        mean_metrics.append(metrics.mean())

    magnet_model.train()
    return mean_losses, mean_metrics


def train():
    losses, accuracies = [], []
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
                )
                loss /= tonfig.num_grad_accumalation_steps
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
            losses.append(lossf); accuracies.append(accuracy)

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
    return losses, accuracy


if __name__ == "__main__":
    # dataset
    assert "cuda" in tonfig.device, "Only cuda is supported for training."
    dataset = ioPathTextDs(
        save_path=AUDIO_TXT_PATH,
        batch_size=tonfig.batch_size,
        split_float=0.9
    )
    X_train, y_train = dataset["train"]
    X_val, y_val = dataset["val"]

    # for float16 or bfloat16 training
    ctx = torch.autocast(
                device_type=tonfig.device_type,
                dtype={"bfloat16": torch.bfloat16,
                       "float16" : torch.float16}[tonfig.dtype]
            )

    preprocess_ops = PreProOps(
        max_sec=MAX_SEC, 
        device=tonfig.device,
        autocast=ctx,
        compile=True,
        print_info=True
    )
    
    iterator = lambda X, y: iter(
        PreProDataset(
            X, y, audio_pad_id=tonfig.mask_id, 
            wavlen=preprocess_ops.WAVLEN, device=tonfig.device
        ).iter_batches()
    )
    train_iterator, val_iterator = iterator(X_train, y_train), iterator(X_val, y_val)
    # lr scheduler
    get_lr = CosineDecayWithWarmup(
        warmup_steps=tonfig.warmup_steps,
        max_learning_rate=tonfig.max_learning_rate,
        decay_steps=tonfig.decay_steps,
        min_learning_rate=tonfig.min_learning_rate
    )

    os.makedirs(tonfig.ckpt_dir, exist_ok=True)
    if "scratch" in tonfig.init_from:
        model_config = monfig
        model_args = dc.asdict(monfig)
        magnet_model = MAGNET(
            model=Transformer(model_config),
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
            model=Transformer(model_config),
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

    print("Compiling magnet model...")
    magnet_model = torch.compile(magnet_model)
    print("Done.")

    magnet_trainer = MagnetTrainer(magnet_model=magnet_model, preprocess_ops=preprocess_ops, config=tonfig)

    print("Training to be started...")
    losses, accuracies = train()
    plt.title("Loss and Accuracy")
    plt.plot({"losses": losses, "accuracies": accuracies})
    plt.xticks(range(0, tonfig.num_steps, 5000))
    plt.grid(True)
    plt.ylabel("Loss/Accuracy")
    plt.xlabel("Steps")
    plt.show()
    plt.savefig("images/loss_accuracy.png")
    print("Training done.")
