import requests
from tqdm import tqdm
import zipfile
import os

def download(url, save_path=None):
    if save_path is None:
        save_path = url.split("/")[-1]

    response = requests.get(url, stream=True)
    response.raise_for_status()

    file_size = int(response.headers.get('Content-Length', 0))

    with open(save_path, 'wb') as file, tqdm(
        total=file_size,
        unit='B',
        unit_scale=True,
        desc=f'Downloading {save_path}'
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress_bar.update(len(chunk))

    print(f"Downloaded {save_path} ({file_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    data_url = 'http://images.cocodataset.org/zips/train2014.zip'
    data_save_path = 'train2014.zip'

    annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    annotation_save_path = 'annotations_trainval2014.zip'

    download(data_url, data_save_path)
    download(annotation_url, annotation_save_path)
    print("Download completed.")

    # Unzip the downloaded files
    print("Unzipping files...")
    with zipfile.ZipFile(data_save_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"Extracted {data_save_path} to 'train2014' directory.")
    with zipfile.ZipFile(annotation_save_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"Extracted {annotation_save_path} to 'annotations' directory.")