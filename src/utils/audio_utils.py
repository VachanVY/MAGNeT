import matplotlib.pyplot as plt
import tempfile
import os
from typing import Optional

from torch import Tensor
import torchaudio

def playAudio(
        *,
        filepath:Optional[str]=None,
        tensor:Optional[Tensor]=None,
        text:Optional[str]=None,
        save:bool=False,
        sample_rate:int=16_000,
        show:bool=False
    ):
    assert any([filepath is not None, tensor is not None]), "provide either wav-path or a wav-tensor of shape 1"
    if text is not None:
        print("Prompt:", text, sep="\n")
    if filepath:
        assert filepath.endswith(".wav")
        if show:
            wav = torchaudio.load(filepath)[0][0]
            plt.plot(wav.tolist()); plt.show()
        os.system(f'aplay {filepath}')
    elif tensor is not None:
        tensor = tensor.cpu().float()
        if save:
            assert filepath is not None, "provide filepath if you want to save"
            torchaudio.save(filepath, src=tensor)
            playAudio(filepath=filepath, tensor=None, show=show)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                torchaudio.save(tmp.name, src=tensor.unsqueeze(0), sample_rate=sample_rate)
                playAudio(filepath=tmp.name, tensor=None, show=show)
                