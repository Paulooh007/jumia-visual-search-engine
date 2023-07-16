import streamlit as st
import pinecone
from dotenv import load_dotenv
from pathlib import Path
import os
from image_search_engine import utils
from image_search_engine.product_image_search import JumiaProductSearch
from PIL import Image, ImageOps

import requests


from streamlit_image_select import image_select

jumia = JumiaProductSearch()


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
            images=[
                "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/88/3305912/1.jpg?5807",
                "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/74/0464341/1.jpg?7325",
                "https://watchlocker.ng/wp-content/uploads/2021/04/JY8085-14H.jpg",
                "https://www-konga-com-res.cloudinary.com/w_auto,f_auto,fl_lossy,dpr_auto,q_auto/media/catalog/product/M/L/196920_1641394875.jpg",
                "https://www-konga-com-res.cloudinary.com/w_auto,f_auto,fl_lossy,dpr_auto,q_auto/media/catalog/product/I/K/154983_1595624114.jpg",
                "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/73/3254702/1.jpg?5592",
                "https://store.storeimages.cdn-apple.com/4668/as-images.apple.com/is/MKUQ3_VW_34FR+watch-44-alum-midnight-cell-se_VW_34FR_WF_CO_GEO_AE?wid=1400&hei=1400&trim=1%2C0&fmt=p-jpg&qlt=95&.v=1683237043713",
                "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/71/6579011/1.jpg?5730",
            ],
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
