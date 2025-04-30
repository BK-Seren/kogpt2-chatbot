# chatbot_app.py

import streamlit as st

# 1. 페이지 설정 (가장 위에!)
st.set_page_config(page_title="딥리드 어휘 챗봇", layout="centered")

from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch



# 2. KoGPT2 모델 불러오기
@st.cache_resource
def load_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    return tokenizer, model

tokenizer, model = load_model()

# 3. 사전 검색 함수
def search_korean_dictionary(word):
    encoded = quote(word)
    url = f"https://stdict.korean.go.kr/search/searchResult.do?pageSize=1&searchKeyword={encoded}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    title = soup.select_one(".search_result dt a")
    definition = soup.select_one(".search_result dd span")

    if title and definition:
        return f"📘 **{title.text.strip()}**: {definition.text.strip()}"
    else:
        return "❌ 사전에서 정의를 찾을 수 없습니다."

# 4. KoGPT2 예문 생성
def generate_example_sentence(word):
    prompt = f"'{word}'이라는 단어를 포함한 쉬운 예문을 만들어줘."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=80, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

# 5. UI 구성
st.title("🧠 딥리드 어휘 챗봇")
st.caption("KoGPT2 + 표준국어대사전 API 기반")

user_input = st.text_input("어휘 또는 문장을 입력해보세요:")

if st.button("질문하기") and user_input.strip():
    with st.spinner("답변 생성 중..."):
        target_word = user_input.strip().split()[0]  # 첫 단어 기준
        
        dict_result = search_korean_dictionary(target_word)
        example_sentence = generate_example_sentence(target_word)

        st.markdown("### 📘 사전 정의")
        st.markdown(dict_result)

        st.markdown("### ✍️ 예문 생성")
        st.markdown(f"\"{example_sentence}\"")
