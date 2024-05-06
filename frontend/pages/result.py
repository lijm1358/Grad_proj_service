# import os

# import requests
import streamlit as st
from pyparsing import empty
from utils import show_img

st.set_page_config(layout="wide")
empty1, con0, empty2 = st.columns([0.1, 0.15, 0.1])
empty1, con1, empty2 = st.columns([0.1, 0.3, 0.1])
empty1, line, empty2 = st.columns([0.1, 0.3, 0.1])
empty1, _, con2, empty2 = st.columns([0.2, 1.5, 0.2, 0.2])  # two button


def main():
    with empty1:
        empty()
    print(st.session_state.item_select)
    # datas = {"image_id": st.session_state.item_select["item_id"], "user_id": st.session_state.user_id}
    # response = requests.post(os.environ["url"] + "interacte", data=datas).json()
    # if response.status_code == 200:
    with con0:
        show_img(st.session_state.item_select["item_id"], st.session_state.item_select["image_link"])
    with con1:
        st.write(f"상품 번호 : {st.session_state.item_select['item_id']}")
        st.write(f"상품 이름 : {st.session_state.item_select['prod_name']}")
        st.write(f"상품 분류 : {st.session_state.item_select['prod_type_name']}")
        st.write(f"상품 설명 : {st.session_state.item_select['detail_desc']}")
    with line:
        st.write()
        st.write()
        st.write()
    with con2:
        st.session_state.prompt = ""
        st.session_state.img_select = ""
        st.session_state.img_gen = 1
        st.page_link(page="./app.py", label="처음으로", icon="🏠")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
