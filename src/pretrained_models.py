import torch
import torchaudio

from encodec import (
    EncodecModel,
    utils as enc_utils
)
from transformers import AutoTokenizer, T5EncoderModel


encodec_model = EncodecModel.encodec_model_24khz() # in eval mode
encodec_model.set_target_bandwidth(3.0) # 3 kbps (Nq = 4)
SRATE = encodec_model.sample_rate
MAX_SEC = 10
WAVLEN = SRATE*MAX_SEC

## (google-t5/t5-small) (google-t5/t5-base) (google-t5/t5-large) (google-t5/t5-3b) (google-t5/t5-11b)
T5_MODEL_PATH = "google-t5/t5-small"
cond_model = T5EncoderModel.from_pretrained(T5_MODEL_PATH) # in eval mode
cond_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)


def preprocess_wavpath(wavpath:str, target_srate:int=SRATE):    
    wav, sr = torchaudio.load(wavpath)
    wav = enc_utils.convert_audio(
            wav, sr=sr, target_sr=target_srate,
            target_channels=encodec_model.channels
        )
    return wav


def get_conditioned_tensor(padded_cond_seq:list[str]) -> torch.Tensor:
    with torch.no_grad():
        cond_tensors = cond_model(
            input_ids=padded_cond_seq["input_ids"], # (B, N)
            attention_mask=padded_cond_seq["attention_mask"] # (B, N)
        ) # (B, N, cond_dim)
    return cond_tensors
