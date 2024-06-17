import torch
import torchaudio

from encodec import (
    EncodecModel,
    utils as enc_utils
)
from transformers import (
    AutoTokenizer, T5EncoderModel
)

#---------------------------------------------------------------------------------------------------#
## (google-t5/t5-small) (google-t5/t5-base) (google-t5/t5-large) (google-t5/t5-3b) (google-t5/t5-11b)
T5_MODEL_PATH = "google-t5/t5-small"
#---------------------------------------------------------------------------------------------------#

class PreProOps:
    def __init__(self, max_sec:float|int, print_info:bool=False, device:str="cuda"):
        
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(3.0) # 3 kbps (Nq = 4)

        self.SRATE = self.encodec_model.sample_rate
        self.WAVLEN = int(self.SRATE*max_sec)

        if print_info: print("compiling encodec model...")
        self.encodec_model.to(device)
        self.encodec_model = torch.compile(self.encodec_model)
        if print_info: print("compiled encodec model.")
        self.encodec_model.eval() # in eval mode
        
        self.cond_model = T5EncoderModel.from_pretrained(T5_MODEL_PATH)
        if print_info: print("compiling T5 model...")
        self.cond_model.to(device)
        self.cond_model = torch.compile(self.cond_model)
        if print_info: print("compiled T5 model.")
        self.cond_model.eval() # in eval mode

        self.cond_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
        
        
    def preprocess_wavpath(self, wavpath:str, squeeze:bool=True) -> torch.Tensor:
        wav, sr = torchaudio.load(wavpath)
        wav = enc_utils.convert_audio(
                wav, sr=sr, target_sr=self.SRATE,
                target_channels=self.encodec_model.channels
            )
        if squeeze:
            return wav.squeeze()
        return wav
    
    def getQuantizedCodings(self, bs_waves:torch.Tensor) -> torch.Tensor: # (B, C=1, T=WAVLEN)
        with torch.no_grad():
            encoded_frames = self.encodec_model.encode(bs_waves)
        codes = encoded_frames[0][0] # (B, n_q=4, T=750)
        return codes
    
    def getAudioFromCodings(self, codes:torch.Tensor):
        encoded_frames = [(codes, None)]
        with torch.no_grad():
            wavs = self.encodec_model.decode(encoded_frames)
        return wavs
    
    def get_conditioned_tensor(self, padded_cond_seq:list[str]) -> torch.Tensor:
        with torch.no_grad():
            cond_tensors = self.cond_model(
                input_ids=padded_cond_seq["input_ids"], # (B, N)
                attention_mask=padded_cond_seq["attention_mask"] # (B, N)
            )
        return cond_tensors.last_hidden_state # (B, N, cond_dim)
    
    def tokenize(self, text_str:str, padding:bool=True) -> torch.Tensor:
        return self.cond_tokenizer(text_str, return_tensors="pt", padding=padding)
