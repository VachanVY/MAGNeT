import requests
from tqdm import tqdm
import os; import glob; import shutil
import random as r
import typing as tp
import pickle
from itertools import islice
import json


#----------------------------------------------------------------------------------------------#
MAX_SEC = 10
QCODING_LEN = 750
DATA_DIR = "data/MusicBench"
URLS = [
    "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench_train.json"
    "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench_test_A.json",
    "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench_test_B.json",
    "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz",
]
FILE_PATHS:list[str] = [os.path.join(DATA_DIR, url.split("/")[-1]) for url in URLS]

DATA_PATH = os.path.join(DATA_DIR, "datashare/data_aug2")
AUDIO_TXT_PATH = os.path.join(DATA_DIR, "audiopath_txt.pkl")
#----------------------------------------------------------------------------------------------#


def shuffle_preserve_order(a, b):
    combined = list(zip(a, b))
    r.shuffle(combined)
    
    a, b = zip(*combined)
    return a, b

def get_shape(lst):
    if not isinstance(lst, list):
        return []
    
    shape = []
    current_level = lst
    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) == 0:
            break
        current_level = current_level[0]
    
    return shape


def split_ds(x, y, split_float:float):
    # assert len(x) == len(y), f"len(x): {len(x)} and len(y): {len(y)}"
    train_len = int(split_float*len(x))

    x_train, y_train = x[:train_len], y[:train_len]
    x_val, y_val = x[train_len:], y[train_len:]
    
    assert get_shape(x_train) == get_shape(y_train)
    assert get_shape(x_val) == get_shape(y_val)
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val)
    }


def download_file(url:str, filename:str, chunk_size:int=1024):
    """Download a file from the given URL and save it with the specified filename"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    print(f"{filename} downloaded successfully:")


def download_dataset():
    "download if data dir is empty"
    os.makedirs(DATA_DIR, exist_ok=True)

    # if files not present in DATA_DIR then start downloading
    if not os.listdir(DATA_DIR):
        # download files
        for url, file_path in zip(URLS, FILE_PATHS):
            # if that file doesn't exist then only downloading else skip downloading
            if not os.path.exists(file_path):
                try:
                    # the tar file is very big so may freeze download in-between. 
                    download_file(url, file_path)
                except: # KeyboardInterrupt download freezes
                    print(f"Couldn't download {file_path}. So deleting that file, try again. Maybe try manually downloading it")
                    os.remove(file_path)
                # if file is a .tar.gz type unpack it
                if file_path.endswith(".tar.gz"):
                    print(f"Unpacking {file_path}")
                    return_code = os.system(f"tar -xzf {file_path} -C {DATA_DIR}")
                    print(
                        "Unpacking succesful," if return_code==0 else 
                        f"Error code {return_code} when unpacking {file_path}"
                    )
            else:
                print(f"{file_path} is present, skipping downloading it.")
        
        # move unpacked files from diff folder to the one folder and delete src folder
        src_dir = os.path.join(DATA_DIR, "datashare/data")
        for fpath in os.listdir(src_dir):
            fpath = os.path.join(src_dir, fpath)
            shutil.move(src=fpath, dst=DATA_PATH)
        print(f"Moved files from {src_dir} to {DATA_PATH} so deleting {src_dir}")
        os.removedirs(src_dir)
        return
    print(
        f"Data already present, skipping downloading. If you want to redownload then delete {DATA_DIR} folder and run again.\n"
    )


def ioPathTextDs(
    save_path:str,
    batch_size:tp.Union[int, None]=None,
    split_float:float = 0.9,
    return_ds:bool=False,
    force_redo:bool=False
):
    if not os.path.exists(save_path) or force_redo:
        print("Saving dataset...")
        assert batch_size is not None, "provide batch_size when saving."
        paths, texts = [], []
        with open(os.path.join(DATA_DIR, 'MusicBench_train_modified.json')) as json_data:
            for line in json_data:
                # convert "string-dictionary" to dictionary
                data:dict[str, str] = json.loads(line.replace("'", "/"))
                # data dictionary with music file path and text caption for that music             
                wavpath, text = (data["location"], data["main_caption"])
                if not wavpath.split("/")[0].endswith("_aug2"):
                    dirname, filename = wavpath.split("/")
                    wavpath = os.path.join(dirname + "_aug2", filename)
                wavpath = os.path.join(DATA_DIR, "datashare", wavpath)
                paths.append(wavpath); texts.append(text)
        assert len(paths)==len(texts), "Number of audio paths must be equal to number of captions"
        # shuffle dataset
        paths, texts = shuffle_preserve_order(paths, texts)
        paths = [list(islice(paths, i, i+batch_size)) for i in range(0, len(paths), batch_size)][:-1] # (len//batch_size, batch_size)
        texts = [list(islice(texts, i, i+batch_size)) for i in range(0, len(texts), batch_size)][:-1] # (len//batch_size, batch_size)
        # save dataset
        with open(save_path, "wb") as file:
            pickle.dump(
                obj=[paths, texts], file=file,
                protocol=pickle.HIGHEST_PROTOCOL
            )
    else:
        # load dataset
        with open(save_path, "rb") as file:
            paths, texts = pickle.load(file)
        if len(paths[0]) != batch_size:
            print(f"Making dataset for batch_size: {batch_size} instead of `batch_size` in saved dataset.")
            paths, texts = ioPathTextDs(
                AUDIO_TXT_PATH, 
                batch_size=batch_size, 
                split_float=split_float,
                return_ds=True,
                force_redo=True
            )
        print("Dataset is preprocessed.")
    if return_ds:
        return split_ds(paths, texts, split_float=split_float)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', default=False, type=bool)
    parser.add_argument('--preprocess', default=False, type=bool)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    if args.download:
        print("Request: --download")
        download_dataset()
    if args.preprocess:
        print("Request: --preprocess")
        ioPathTextDs(AUDIO_TXT_PATH, batch_size=args.batch_size, split_float=0.9)
