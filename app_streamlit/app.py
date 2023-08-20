import requests
import streamlit as st
from PIL import Image, ImageOps
from streamlit_image_select import image_select

from image_search_engine import utils
from image_search_engine.product_image_search import JumiaProductSearch

jumia = JumiaProductSearch()

file_path = "app_streamlit/image_urls.txt"
url_list = []
with open(file_path, "r") as file:
    for line in file:
        url = line.strip()
        url_list.append(url)


def get_search_results(query):
    res = jumia.search(query, 9)

    images, names, urls = [], [], []

    for i, record in enumerate(res["matches"]):
        metadata = record["metadata"]
        images.append(metadata["product_image_url"])
        names.append(metadata["product_name"])
        urls.append(metadata["product_url"])

    return images, names, urls


banner_img = Image.open(utils.PACKAGE_DIR.parent / "jumia_lens.png")
st.image(banner_img)


input_options = st.radio("Select Input Option", ("image upload", "use example images"))

img = None

if input_options == "image upload":
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


else:
    with st.expander(label="Chose sample image", expanded=False):
        img = image_select(
            label="Use example image",
            images=url_list,
        )


if img:
    if isinstance(img, str):
        image = Image.open(requests.get(img, stream=True).raw)
    else:
        image = Image.open(img)

    image = ImageOps.exif_transpose(image)

    with st.columns(3)[1]:
        st.markdown("### Query Image.")
        st.image(image)

    n = 3
    product_images, product_names, product_urls = get_search_results(image)

    for i, col in enumerate(st.columns(n)):
        positions = (i, i + 3, i + 6)
        names = [product_names[i] for i in positions]
        images = [product_images[i] for i in positions]
        urls = [product_urls[i] for i in positions]

        with col:
            st.write(names[0])
            st.image(images[0])
            st.write(urls[0])

            st.write(names[1])
            st.image(images[1])
            st.write(urls[1])

            st.write(names[2])
            st.image(images[2])
            st.write(urls[2])

            # st.write(names[3])
            # st.image(images[3])
            # st.write(urls[3])
