# from pathlib import Path
from image_search_engine.metadata import shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "jumia_3650"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "jumia_3650"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "jumia_3650"
