# import os

# import requests
import streamlit as st
from pyparsing import empty
from utils import show_item

st.set_page_config(layout="wide")
empty1, con0, empty2 = st.columns([0.1, 2.0, 0.1])  # item1-5
empty1, con1, con2, con3, con4, con5, empty2 = st.columns([0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1])  # item1-5
empty1, con6, con7, con8, con9, con10, empty2 = st.columns([0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1])  # item6-10
empty1, con11, empty2 = st.columns([0.1, 2.0, 0.1])  # empty limne
empty1, con12, _, con13, empty2 = st.columns([0.1, 0.15, 1.6, 0.25, 0.1])  # two button


temp_data = {
    0: "https://bit.ly/3wcFeXj",
    1: "https://bit.ly/3ULUaos",
    2: "https://bit.ly/3UJTakS",
}


def main():
    with empty1:
        empty()
    # datas = {"image_id": st.session_state.img_select, "user_id": st.session_state.user_id}
    rec_results = None
    # response = requests.post(os.environ["url"] + "recommend", data=datas).json()
    # if response.status_code == 200:
    #     rec_results = response["rec_results"]
    # else:
    rec_results = [
        {
            "item_id": "0108775015" if i % 2 else "1234",
            "prod_name": "Strap top",
            "prod_type_name": "Vest top",
            "detail_desc": "Womens Everyday Basics,1002,Jersey Basic,Jersey top with narrow shoulder straps.",
            "image_link": ("https://bit.ly/3UO6Uey" if i % 4 == 0 else temp_data[i % 3]),
        }
        for i in range(10)
    ]

    with con0:
        st.header(f"'{st.session_state.prompt}' 중 {st.session_state.user_id}님의 취향을 고려한 상품 😎")
    if rec_results is not None:
        with con1:
            show_item(rec_results[0])
        with con2:
            show_item(rec_results[1])
        with con3:
            show_item(rec_results[2])
        with con4:
            show_item(rec_results[3])
        with con5:
            show_item(rec_results[4])
        with con6:
            show_item(rec_results[5])
        with con7:
            show_item(rec_results[6])
        with con8:
            show_item(rec_results[7])
        with con9:
            show_item(rec_results[8])
        with con10:
            show_item(rec_results[9])
        with con11:
            id_list = [item["item_id"] for item in rec_results]
            select_id = st.selectbox(
                "가장 마음에 드는 아이템을 선택하세요!",
                id_list,
            )
            st.session_state.item_select = rec_results[id_list.index(select_id)]
        with con12:
            st.session_state.prompt = ""
            st.session_state.img_select = ""
            st.session_state.img_gen = 1
            st.page_link(page="./app.py", label="처음으로", icon="🏠")
        with con13:
            st.page_link(page="./pages/result.py", label="다음 페이지", icon="➡️")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
