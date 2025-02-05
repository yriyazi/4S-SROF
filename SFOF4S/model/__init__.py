from .opencv_functions      import *
from .edge_superres_pytorch import PyTorchModel

import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def check_and_download():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "converted_model.pt")
    url = "https://raw.githubusercontent.com/yriyazi/SFOF4S/refs/heads/main/SFOF4S/model/converted_model.pt"
    
    if not os.path.exists(filename):
        print(f"{filename} not found. Downloading...")
        download_file(url, filename)
    else:
        print(f"{filename} already exists.")

check_and_download()
