import os

import requests
import streamlit as st
from pyparsing import empty
from utils import show_item

st.set_page_config(layout="wide")
empty1, con0, empty2 = st.columns([0.1, 2.0, 0.1])  # item1-5
empty1, con1, con2, con3, con4, con5, empty2 = st.columns([0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1])  # item1-5
empty1, con6, con7, con8, con9, con10, empty2 = st.columns([0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1])  # item6-10
empty1, con11, empty2 = st.columns([0.1, 2.0, 0.1])  # empty limne
empty1, con12, _, con13, empty2 = st.columns([0.1, 0.15, 1.6, 0.25, 0.1])  # two button
empty1, con14, empty2 = st.columns([0.725, 0.15, 0.725])  # two button


temp_data = {
    0: "https://bit.ly/3wcFeXj",
    1: "https://bit.ly/3ULUaos",
    2: "https://bit.ly/3UJTakS",
}


def main():
    if st.session_state.user_id == "":
        st.switch_page("./app.py")
    with empty1:
        empty()
    if st.session_state.rec_results is None:
        datas = {"image_id": st.session_state.img_select, "user_id": st.session_state.user_id}
        rec_results = None
        response = requests.post(os.environ["url"] + "/api/recommend", json=datas)
        response_json = response.json()
        if response.status_code == 200:
            rec_results = response_json["rec_results"]
            st.session_state.rec_results = rec_results
    else:
        rec_results = st.session_state.rec_results

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
            id_list = [item["article_id"] for item in rec_results]
            select_id = st.selectbox(
                "가장 마음에 드는 아이템을 선택하세요!",
                id_list,
            )
            st.session_state.item_select = rec_results[id_list.index(select_id)]
        with con12:
            st.session_state.prompt = ""
            st.session_state.img_select = ""
            st.session_state.img_gen = 1
            st.page_link(page="./pages/image.py", label="처음으로", icon="🏠")
        with con13:
            st.page_link(page="./pages/result.py", label="다음 페이지", icon="➡️")
        with con14:
            st.header("")
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
