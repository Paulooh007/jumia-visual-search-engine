import zipfile
from pathlib import Path

from tqdm import tqdm


def zip_folder(source_dir, out_dir):
    directory = Path(f"{source_dir}/")

    with zipfile.ZipFile(out_dir, mode="w") as archive:
        for file_path in tqdm(directory.iterdir(), desc="Zipping..."):
            archive.write(file_path, arcname=file_path.name)
