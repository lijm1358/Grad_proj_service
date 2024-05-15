from io import BytesIO

import requests
import streamlit as st
from PIL import Image


@st.cache_resource
def show_img(img_id, img_url):
    image = Image.open(BytesIO(requests.get(img_url, stream=True).content))
    st.image(image, caption=img_id, use_column_width=True)


@st.cache_resource
def show_item(item_info):
    show_img(item_info["article_id"], item_info["image_link"])
    _, temp, _ = st.columns([0.35, 0.5, 0.3])
    with temp:
        with st.popover("info"):
            st.markdown(f"상품 번호 : {item_info['article_id']}")
            st.markdown(f"상품 이름 : {item_info['prod_name']}")
            st.markdown(f"상품 분류 : {item_info['prod_type_name']}")
            st.markdown(f"상품 설명 : {item_info['detail_desc']}")


def init_session_state():
    st.session_state.prompt = ""
    st.session_state.user_id = ""
    st.session_state.img_select = ""
    st.session_state.item_select = {}
    st.session_state.img_gen = 1
    st.session_state.images = None
    st.session_state.rec_results = None
    st.session_state.req_id = ""
