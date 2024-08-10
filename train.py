import typing as tp
import math
import bisect
import random as r
import dataclasses as dc
import os
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
from tqdm import trange

import torch; torch.cuda.empty_cache()
from torch import nn, Tensor
from torch.nn import functional as F
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" # uncomment this if getting Out Of Memory Error

from src.music_bench import (
    split_ds,
    ioPathTextDs,
    AUDIO_TXT_PATH,
    DATA_DIR,
    QCODING_LEN,
    MAX_SEC,
    PreProDataset
)
from src.preprocess_ops import PreProOps
from src.model import (
    monfig,
    get_magnet_model,
    MAGNET # only for type hinting
)
from src.utils.lr_scheduler import CosineDecayWithWarmup

ONLINE = True # True if conditioned text and qcodings are computed on the go else False
#---------------------------------------------------------------------------------------------------#
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#---------------------------------------------------------------------------------------------------#
@dc.dataclass
class tonfig:
    seed:int = 42

    # general train args
    num_steps:int = 100_000 # 1_000_000 in paper
    batch_size:int = 64
    # number_of_tokens_per_update = batch_size*seqlen*num_grad_accumalation_steps = 64*750*3 = 144000
    # Only one codebook is taken at a time, so all codebooks are not considered for number of tokens per update
    num_grad_accumalation_steps:int = 3
    log_interval:int = 1

    # Early stopping args
    patience:int = 10
    restore_best_weights:bool = True

    # MAGNeT and Model args
    spanlen:int = monfig.spanlen # 3
    ## equal to the maximum length of the quantized codings for 10 seconds of audio
    seqlen:int = QCODING_LEN # 750
    attn_window:int = 5

    # Classifier-Free Guidance
    cfg_dropout:float = 0.1

    # vocab (without mask_id): {0, ..., 1023} => mask_id = 1024
    # cardinality = len(vocab) which is 1024
    mask_id:int = monfig.cardinality

    # cosine decay with warmup args
    warmup_steps:int = 1250
    max_learning_rate:float = 5e-4
    decay_steps:int = num_steps
    min_learning_rate:float = 0.0*max_learning_rate

    # optimizer args
    decay:bool = True # if true uses cosine decay else max_learning_rate thoughtout training
    weight_decay:float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    clipnorm:float = 1e0
    ema_momentum:float = 0.99

    # eval and checkpointing args
    eval_freq:int = 2000
    eval_steps:int = 100
    init_from:str = "scratch" # 'scratch' or 'resume'
    ckpt_dir:str = "ckpts"
    always_checkpoint:bool = False
    assert restore_best_weights != always_checkpoint

    # device args
    device:str = "cuda" # 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
    device_type:str = "cuda" if "cuda" in device else "cpu"

    # dtype args # 'encodec model has LSTMs which doesn't work with bfloat16'
    dtype:str = "float16" # float16

    PRECOMPUTED_TENSORS_DIRPATH:str = os.path.join(DATA_DIR, f"precondition_batch_size_{batch_size}")
    assert os.path.exists(PRECOMPUTED_TENSORS_DIRPATH)
#---------------------------------------------------------------------------------------------------#
RANDGEN:r.Random = r.Random(tonfig.seed)
GENRERATOR:torch.Generator = torch.Generator(device=tonfig.device).manual_seed(tonfig.seed+1)
#---------------------------------------------------------------------------------------------------#

class MagnetTrainer:
    def __init__(
        self, 
        magnet_model:MAGNET, 
        config:tonfig,
    ):
        self.magnet_model = magnet_model
        self.span_len = config.spanlen
        self.mask_id = config.mask_id
        self.device = torch.device(config.device)

        self.generator = GENRERATOR
        self.randgen = RANDGEN

        self.cfg_dropout = config.cfg_dropout
        self._num_spans_to_mask = torch.tensor(
            self._get_number_of_spans_to_mask(T=config.seqlen, L=config.spanlen),
            device=self.device
        ) # (100,)

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
        
        batch_randperm = torch.rand((B, seq_len), device=self.device, generator=self.generator).argsort(-1) # rand integers from 0 to T
        mask = batch_randperm < batched_num_spans[..., None] # contains `batched_num_spans` number of Trues
        shifted_mask = mask.clone()
        for _ in range(self.span_len-1):
            shifted_mask = torch.concat((torch.zeros((B, 1), device=self.device).bool(), shifted_mask[:, :-1]), dim=-1)
            mask = torch.logical_or(mask, shifted_mask)
        return mask # (B, T) # True value will be masked

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

        loss = F.cross_entropy(
            y_pred.movedim(-1, -2), # (B, num_classes=cardinatlity, d1=T) <= (B, T, cardinality)
            y_true, # (B, d1=T)
            ignore_index=-1
        )
        return loss

    def mini_train_step( # mini train step if gradient accumulation is used
        self,
        audio_input:dict[str, Tensor],
        cond_tensor:Tensor,
    ):
        """```
        audio_input: {"qcode": audio_tokens,   # tensor of shape (B, Nq, T)
                      "mask": audio_pad_mask} # tensor of shape (B, Nq, T)
        cond_str: batch(text)
        ```"""
        # audio tensor and mask
        (
            audio_tokens,
            audio_pad_mask # False value contains the padded tokens # True value is to be taken # loss taken on True values
        ) = audio_input["qcode"], audio_input["mask"]

        # Batch_dim, Codebooks, Time_dim
        B, Nq, T = audio_tokens.shape

        # conditioning tensor
        cond_tensor = self.magnet_model.cfg(
            cond_tensor=cond_tensor,
            randf=self.randgen.random(),
            cfg_dropout=self.cfg_dropout,
        )

        # A random phase: for masking a random codebook
        phase = self.randgen.randint(0, Nq-1)
        
        # masking rate = gamma(i, s) = cos(pi*(i-1)/(2*s)) where i is the decoding time step i ϵ [1, s]
        # during training time step i is randomly sampled, thus (i-1)/s is randomly sampled which is `rand_decoding_time`
        rand_decoding_time = torch.rand(B, device=self.device, generator=self.generator) # (B,)
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
        # (B, Nq, T) =sum(-1, -2)> (B,) =mean=> ()
        accuracy = self.get_accuracy(audio_tokens, logits.argmax(-1), loss_mask=loss_mask)
        return loss, accuracy
    
    @staticmethod
    @torch.no_grad()
    def get_accuracy(
        y_true:Tensor, # (B, Nq, T)
        y_pred:Tensor, # (B, Nq, T, cardinality)
        loss_mask:tp.Optional[Tensor]=None # (B, Nq, T)
    ):
        y_true, y_pred = y_true.flatten(1), y_pred.flatten(1) # (B, Nq*T)
        corr_bool = (y_true == y_pred)
        num = y_true.shape[-1]
        acc = (corr_bool.sum(-1)/num).mean().item()

        if loss_mask is not None:
            loss_mask = loss_mask.flatten(1)
            batch_mask_acc = list(map(MagnetTrainer._masked_accuracy, y_true, y_pred, loss_mask))
            return acc, sum(batch_mask_acc)/len(batch_mask_acc)
        return acc
    
    def _masked_accuracy(
        y_true:Tensor, # (Nq*T)
        y_pred:Tensor, # (Nq*T)
        mask:Tensor    # (Nq*T)
    ):
        corr_bool = y_true == y_pred
        num = mask.sum(-1)
        acc = corr_bool[mask].sum(-1)/num
        return acc.item()


@torch.no_grad()
def update_ema(ema_model:MAGNET, model:MAGNET, decay:float):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # ema = decay*ema + (1-decay)*param
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


@torch.no_grad()
def evaluate():
    print("Evaluating...")
    magnet_model.eval()

    mean_losses, mean_metrics = [], []
    for eval_iterator in [train_iterator, val_iterator]:
        losses = torch.empty((tonfig.eval_steps,), device=tonfig.device)
        metrics = torch.empty_like(losses, device=tonfig.device)

        for eval_step in trange(tonfig.eval_steps):
            audio_input, cond_tensor = next(eval_iterator)
            with ctx:
                loss, acc = magnet_trainer.mini_train_step(
                    audio_input=audio_input,
                    cond_tensor=cond_tensor
                )
            losses[eval_step], metrics[eval_step] = loss, acc[0]
        mean_losses.append(losses.mean())
        mean_metrics.append(metrics.mean())

    magnet_model.train()
    return mean_losses, mean_metrics


def train(losses:list=[], accuracies:list=[], mask_acc:list=[]):
    global best_val_loss, start_iter
    audio_input, cond_tensor = next(train_iterator)

    print("Training about to start...")
    t0 = time.time()
    wait = 0; best_step = 0 # for early stopping
    for step in range(start_iter, tonfig.num_steps):
        # evaluate
        if step % tonfig.eval_freq == 0 and step > start_iter:
            mean_losses, mean_accuracies = evaluate()
            print(
                    f"\t| Training Loss: {mean_losses[0]:.4f} || Training Accuracy (Not Masked): {mean_accuracies[0]:.4f} |" 
                    f"| Validation Loss: {mean_losses[1]:.4f} || Validation Accuracy (Not Masked): {mean_accuracies[1]:.4f} |"
                )
            # early stopping
            if mean_losses[1] < best_val_loss or tonfig.always_checkpoint:
                best_val_loss = mean_losses[1]
                wait = 0; best_step = step

                checkpoint = {
                    "model_state": magnet_model.state_dict(),
                    "ema_state": ema.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_args": model_args,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "train_config": dc.asdict(tonfig()),
                    "losses": losses,
                    "accuracies": accuracies,
                    "mask_accuracies": mask_acc
                }
                print(f"Saving checkpoint to {tonfig.ckpt_dir} ...", end=" ==> ")
                torch.save(checkpoint, os.path.join(tonfig.ckpt_dir, "ckpt.pt"))
                print("Done.")
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
                    audio_input=audio_input, cond_tensor=cond_tensor
                )
                if loss.isnan() or loss.isinf():
                    raise RuntimeError(f"Step: {step}\nMini-Step: {mini_step}\nLoss: {loss}\nDIVERGING :(")
                loss /= tonfig.num_grad_accumalation_steps
                # async prefetch immediately
                audio_input, cond_tensor = audio_input, cond_tensor

            # keeps on scaling and adding gradients when calling backward
            scaler.scale(loss).backward()

        if tonfig.clipnorm is not None:
            # unscale the gradients
            scaler.unscale_(optimizer)
            # clips gradients in-place to grad norm
            norm = nn.utils.clip_grad_norm_(magnet_model.parameters(), max_norm=tonfig.clipnorm, error_if_nonfinite=True)

        # calls unscale to the optimizer unless already called, checks for infs and nans as a part of unscale_
        # calls optimizer.step on unscaled grads if no infs and nans else optimizer.step is skipped
        scaler.step(optimizer)

        # update ema
        update_ema(ema, magnet_model, tonfig.ema_momentum)

        # Update the scale factor
        scaler.update()

        # flush grads to save memory
        optimizer.zero_grad(set_to_none=True)

        # some logging
        t1 = time.time()
        dt = t1-t0
        t0 = t1
        if step % tonfig.log_interval == 0:
            # multiply as loss was scaled for gradient accumulation
            lossf = loss.item() * tonfig.num_grad_accumalation_steps
            print(
                f"| Step: {step} || Loss: {lossf:.4f} || Masked Accuracy: {accuracy[1]:.4f} | Accuracy: {accuracy[0]:.4f} |"
                f"| LR: {lr:e} || dt: {dt*1000:.2f}ms |"
                f"| Norm: {norm:.4f} |" if tonfig.clipnorm is not None else "" # if norm is not stable, something is wrong
            )
            losses.append(lossf); accuracies.append(accuracy[0]); mask_acc.append(accuracy[1])

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
        magnet_model.load_state_dict(model_state) # load state maybe for eval after training, although not needed
    return losses, accuracies, mask_acc


if __name__ == "__main__":
    # dataset
    assert "cuda" in tonfig.device, "Only cuda is supported for training."

    # for float16 or bfloat16 training
    ctx = torch.autocast(
        device_type=tonfig.device_type,
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[tonfig.dtype]
    )
    
    X_train, y_train, X_val, y_val = None, None, None, None
    preprocess_ops = None
    if ONLINE:
        paths, texts = ioPathTextDs(
            save_path=AUDIO_TXT_PATH,
            batch_size=tonfig.batch_size,
            split_float=0.9,
            return_ds=True
        )
        dataset = split_ds(paths, texts, 0.9)
        X_train, y_train = dataset["train"]
        X_val, y_val = dataset["val"]
        del dataset

        preprocess_ops = PreProOps(
            max_sec=MAX_SEC,
            device=tonfig.device,
            autocast=ctx,
            compile=True,
            print_info=False,
        )
    
    iterator = lambda X, y, split: iter(
        PreProDataset(
            split=split,
            randgen=RANDGEN,
            audio_pad_id=tonfig.mask_id,
            qcoding_len=tonfig.seqlen,
            device=tonfig.device,
            pre_computed_tensors_dirpath=tonfig.PRECOMPUTED_TENSORS_DIRPATH,
            online=ONLINE,
            wav_paths=X, texts=y,
            preprocess_ops=preprocess_ops,
        ).iter_batches()
    )
    train_iterator, val_iterator = (
        iterator(X_train, y_train, "train"), 
        iterator(X_val, y_val, "val")
    )
    # lr scheduler
    get_lr = CosineDecayWithWarmup(
        warmup_steps=tonfig.warmup_steps,
        max_learning_rate=tonfig.max_learning_rate,
        decay_steps=tonfig.decay_steps,
        min_learning_rate=tonfig.min_learning_rate
    )

    os.makedirs(tonfig.ckpt_dir, exist_ok=True)
    losses, accuracies, mask_accuracies, start_iter = [], [], [], 0
    if "scratch" in tonfig.init_from:
        model_config = monfig
        model_args = dc.asdict(monfig())
        magnet_model = get_magnet_model(compile=True, monfig=model_config)
        ema = deepcopy(magnet_model)

        train_from_step = 0
        best_val_loss = 1e9
        checkpoint = None
    elif "resume" in tonfig.init_from:
        print("Resuming training using checkpoint...")
        # TODO/NOTE: Is this needed?
        def get_model_state(state_dict:dict[str, Tensor]):
            unwanted_prefix = "_orig_mod." # this prefix gets added when a Model is compiled using torch.compile
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            return state_dict
                
        ckpt_path = os.path.join(tonfig.ckpt_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=tonfig.device)

        model_args = checkpoint["model_args"]
        model_config = monfig(**model_args)
        start_iter = checkpoint["step"]

        magnet_model = get_magnet_model(compile=True, monfig=model_config)
        ema = deepcopy(magnet_model)
        
        magnet_model.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])

        train_from_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        losses, accuracies, mask_accuracies = checkpoint["losses"], checkpoint["accuracies"], checkpoint["mask_accuracies"]

    magnet_model.to(tonfig.device); magnet_model.train()
    ema.to(tonfig.device); ema.eval() # use ema for sampling
    ema.requires_grad_(False)
    update_ema(ema, magnet_model, 0.0) # copy the weights
    print("\nNumber of Parameters in MAGNET Model:", 
            sum(p.numel() for p in magnet_model.parameters() if p.requires_grad)/1e6, "Million Parameters\n"
        )

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

    magnet_trainer = MagnetTrainer(magnet_model=magnet_model, config=tonfig)

    losses, accuracies, mask_accuracies = train(losses, accuracies, mask_accuracies)
    plt.title("Loss and Accuracy")
    plt.plot(losses, label="Loss")
    plt.plot(accuracies, label="Accuracy")
    plt.plot(mask_accuracies, label="Masked Accuracy")
    plt.xticks(range(0, tonfig.num_steps, int(tonfig.num_steps//33.33333)), rotation=90)
    plt.grid(True)
    plt.ylabel("Loss/Accuracy")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig("images/loss_accuracy.png")
    plt.show()
    print("Training done.")
