# chatbot_app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch

# ───────────────────────────────
# KoGPT2 모델 불러오기
@st.cache_resource
def load_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    return tokenizer, model

tokenizer, model = load_model()

def generate_kogpt2_reply(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ───────────────────────────────
# 표준국어대사전 검색 함수
def search_korean_dictionary(word):
    url = f"https://stdict.korean.go.kr/search/searchResult.do?pageSize=1&searchKeyword={word}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    meaning_tag = soup.select_one(".list_search_result .search_result .tit_search a")
    
    if meaning_tag:
        return f"📘 사전 정의: {meaning_tag.text.strip()}"
    else:
        return "❌ 사전에서 정의를 찾을 수 없습니다."

# ───────────────────────────────
# Streamlit UI
st.set_page_config(page_title="딥리드 어휘 챗봇", layout="centered")
st.title("🧠 딥리드 어휘 챗봇")
st.caption("KoGPT2 + 표준국어대사전 API 기반")

user_input = st.text_input("어휘 또는 문장을 입력해보세요:")

if st.button("질문하기") and user_input.strip():
    with st.spinner("답변 생성 중..."):
        # KoGPT2 답변 생성
        kogpt2_reply = generate_kogpt2_reply(user_input)

        # 첫 번째 단어 기준 사전 검색 (간단한 처리)
        target_word = user_input.strip().split()[0]
        dict_result = search_korean_dictionary(target_word)

        # 출력
        st.markdown(f"**🤖 챗봇 응답:** {kogpt2_reply}")
        st.markdown(f"**{dict_result}**")
