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
    def __init__(
        self,
        max_sec:float|int,
        autocast:torch.autocast,
        print_info:bool=False,
        compile:bool=True,
        device:str="cuda",
    ):
        self.device = torch.device(device)
        self.autocast = autocast
        
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(3.0) # 3 kbps (Nq = 4)

        self.SRATE = self.encodec_model.sample_rate
        self.WAVLEN = int(self.SRATE*max_sec)

        self.encodec_model.to(self.device, dtype=torch.float32)
        if compile:
            if print_info: print("compiling encodec model...", end=" => ")
            self.encodec_model.compile()
            if print_info: print("Done.")
        self.encodec_model.eval()
        
        self.cond_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
        self.cond_model = T5EncoderModel.from_pretrained(T5_MODEL_PATH)
        self.cond_model.to(self.device, dtype=torch.float32)
        if compile:
            if print_info: print("compiling T5 model...", end=" => ")
            self.cond_model.compile()
            if print_info: print("Done")
        self.cond_model.eval()

        print("\nNumber of Parameters in Encodec Model:", 
            sum(p.numel() for p in self.encodec_model.parameters() if p.requires_grad)/1e6, "Million Parameters\n"
        )
        print(f"\nNumber of Parameters in T5 Model ({T5_MODEL_PATH}):", 
            sum(p.numel() for p in self.cond_model.parameters() if p.requires_grad)/1e6, "Million Parameters\n"
        )
        
    def preprocess_wavpath(self, wavpath:str, get_qcodings:bool) -> torch.Tensor:
        wav, sr = torchaudio.load(wavpath)
        wav = enc_utils.convert_audio(
                wav, sr=sr, target_sr=self.SRATE,
                target_channels=self.encodec_model.channels
            )[..., :self.WAVLEN].to(self.device) # (B, T=WAVLEN)
        if get_qcodings:
            return self.getQuantizedCodings(wav.unsqueeze(1))[0].T # add channel dimension
        return wav
    
    def getQuantizedCodings(self, bs_waves:torch.Tensor) -> torch.Tensor: # (B, C=1, T=WAVLEN)
        with torch.no_grad():
            with self.autocast:
                encoded_frames = self.encodec_model.encode(bs_waves.to(self.device))
        codes = encoded_frames[0][0] # (B, n_q=4, T=750)
        return codes
    
    def getAudioFromCodings(self, codes:torch.Tensor):
        codes.to(self.device)
        encoded_frames = [(codes, None)]
        with torch.no_grad():
            with self.autocast:
                wavs = self.encodec_model.decode(encoded_frames)
        return wavs
    
    def get_conditioned_tensor(self, padded_cond_seq:dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            with self.autocast:
                cond_tensors = self.cond_model(
                    input_ids=padded_cond_seq["input_ids"].to(self.device), # (B, N)
                    attention_mask=padded_cond_seq["attention_mask"].to(self.device) # (B, N)
                )
        return cond_tensors.last_hidden_state # (B, N, cond_dim)
    
    def tokenize(self, text_str:list[str], padding:bool=True) -> dict[str, torch.Tensor]:
        return self.cond_tokenizer(text_str, return_tensors="pt", padding=padding)
