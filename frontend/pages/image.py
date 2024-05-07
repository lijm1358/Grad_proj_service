# import os
import time

# import requests
import streamlit as st
from pyparsing import empty
from utils import show_img

st.set_page_config(layout="wide")
empty1, con1, empty2 = st.columns([0.2, 1.2, 0.2])  # title
empty1, con2, empty2 = st.columns([0.2, 1.2, 0.2])  # text input
empty1, con3, con4, con5, empty2 = st.columns([0.2, 0.4, 0.4, 0.4, 0.2])  # image
empty1, con6, empty2 = st.columns([0.2, 1.2, 0.2])  # ment
empty1, con7, _, con8, empty2 = st.columns([0.2, 0.2, 0.85, 0.2, 0.2])  # two button
empty1, con9, empty2 = st.columns([0.725, 0.15, 0.725])


def main():
    print(st.session_state)
    if st.session_state.user_id == "":
        st.switch_page("./app.py")
    with empty1:
        empty()
    with con1:
        st.header(f"{st.session_state.user_id}님 반갑습니다.")
    with con2:
        new = st.text_area("상품에 대한 간략한 설명을 입력해주세요. ( 입력 후, cmd(ctrl)+enter ) ")
        st.session_state.prompt = new.strip() if st.session_state.img_gen else st.session_state.prompt.strip()
    if len(st.session_state.prompt) > 0 and st.session_state.img_gen:
        with con2:
            st.write("이미지를 생성합니다.(최대 1분 소요)")
        time.sleep(5)
        # datas = {"prompt": st.session_state.prompt, "user_id": st.session_state.user_id}
        # response = requests.post(os.environ["url"] + "imaggen", data=datas).json()
        # if response.status_code == 200:
        #     images = response["images"]
        st.session_state.img_gen = 0
    if len(st.session_state.prompt) >= 5:
        images = [
            {"image_id": "img1", "image_url": "https://bit.ly/3JM8yqL"},
            {
                "image_id": "img2",
                "image_url": "https://bit.ly/3JOlMTL",
            },
            {
                "image_id": "img3",
                "image_url": "https://bit.ly/4bI0xPB",
            },
        ]
    elif 0 < len(st.session_state.prompt) < 5:
        images = [
            {
                "image_id": "img2",
                "image_url": "https://bit.ly/3wqL875",
            }
            for _ in range(3)
        ]
    elif len(st.session_state.prompt) == 0:
        images = [
            {
                "image_id": "no_img",
                "image_url": "https://bit.ly/3WonhQ5",
            }
            for _ in range(3)
        ]
        st.session_state.img_gen = 1
    with con3:
        show_img(images[0]["image_id"], images[0]["image_url"])
    with con4:
        show_img(images[1]["image_id"], images[1]["image_url"])
    with con5:
        show_img(images[2]["image_id"], images[2]["image_url"])
    with con6:
        if images[0]["image_id"] != "no_img":
            st.session_state.img_select = st.selectbox(
                f"< {st.session_state.prompt} > 에 대한 결과입니다.  마음에 드는 이미지를 선택하세요.",
                (images[0]["image_id"], images[1]["image_id"], images[2]["image_id"]),
            )
            # st.write(st.session_state.img_select, "선택됨")
        else:
            st.write("상품 설명을 입력하고 이미지를 생성하세요!")
    with con7:
        if images[0]["image_id"] != "no_img":
            if st.button("이미지 재생성"):
                st.session_state.img_gen = 1
                st.rerun()
    with con8:
        if st.session_state.img_select in (images[0]["image_id"], images[1]["image_id"], images[2]["image_id"]):
            st.page_link(page="./pages/recommend.py", label="다음 페이지", icon="➡️")
    with con9:
        st.header("")
        st.header("")
        if st.button("Logout"):
            st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
