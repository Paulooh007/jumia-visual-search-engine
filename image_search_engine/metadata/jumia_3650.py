# from pathlib import Path
from image_search_engine.metadata import shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "jumia_3650"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "jumia_3650"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "jumia_3650"


CLASS_DICT = {
    "backpack": 0,
    "ear_pods": 1,
    "flash_drive": 2,
    "headset": 3,
    "mouse": 4,
    "office_chairs": 5,
    "usb_hub": 6,
    "wrist_watch": 7,
}
