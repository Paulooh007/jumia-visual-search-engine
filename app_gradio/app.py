from image_search_engine.product_image_search import JumiaProductSearch
from image_search_engine.metadata import jumia_3650
from image_search_engine.utils import load_json_file
import gradio as gr


product_json = jumia_3650.PROCESSED_DATA_DIRNAME / "jumia_3650.json"
PRODUCT_IMG_DIR = jumia_3650.PROCESSED_DATA_DIRNAME / "train"

json_data = load_json_file(product_json)
jumia = JumiaProductSearch()


def _get_products(file):
    idxs = jumia.search(file)
    images = [PRODUCT_IMG_DIR / json_data[id]["filename"] for id in idxs[0]]

    return images


with gr.Blocks() as demo:
    input_image = gr.Image(label="Upload image file", type="filepath")
    outputs = gr.Gallery(preview=True).style(
        columns=[3], rows=[3], height="auto", width="auto"
    )
    input_image.upload(_get_products, inputs=input_image, outputs=outputs)

demo.launch()
