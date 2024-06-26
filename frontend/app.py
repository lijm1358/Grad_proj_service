# import os

# import requests
import streamlit as st
from dotenv import load_dotenv
from pyparsing import empty
from utils import init_session_state

st.set_page_config(layout="wide")
empty1, con1, empty2 = st.columns([0.2, 1.2, 0.2])  # title
empty1, con2, empty2 = st.columns([0.2, 1.2, 0.2])  # text input
empty1, con3, empty2 = st.columns([0.2, 1.2, 0.2])  # text input
empty1, con4, empty2 = st.columns([0.2, 1.2, 0.2])  # text input

sample_user = {"id": "123", "password": "123"}


def main():
    init_session_state()
    print(st.session_state)

    load_dotenv()

    with empty1:
        empty()
    with con1:
        st.header("Login")
    with con2:
        st.session_state.user_id = st.text_input("Username")
        password = st.text_input("Password")
    with con3:
        if st.button("Login"):
            if st.session_state.user_id == sample_user["id"] and password == sample_user["password"]:
                st.switch_page("./pages/image.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
