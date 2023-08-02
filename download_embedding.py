import os
import requests
import zipfile
import shutil
from tqdm import tqdm

url = "http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"

response = requests.get(url, stream=True)
total_size_in_bytes= int(response.headers.get('Content-Length', 0))
block_size = 1024 #1 Kbyte
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
filename = url.split("/")[-1]

print(f'Downloading {filename}...')
with open(filename, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()

if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    print("ERROR, something went wrong while downloading file")

print('Unzipping file...')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('.')

print('Moving file to the glove folder...')
os.makedirs('glove', exist_ok=True)
shutil.move('glove.42B.300d.txt', 'glove/glove.42B.300d.txt')

print('All tasks completed!')