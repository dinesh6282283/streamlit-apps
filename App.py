import streamlit as st
from transformers import pipeline
st.title("Sentiment Analysis ")
p = pipeline(model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student")


message=st.text_area("Please Enter your text")
if st.button("Analyze the Sentiment"):
    mod = p(message)
    result = mod[0]['label'].upper()
    if result == "POSITIVE":
        st.warning("The predicted sentiment is positive!!")
    elif result=="NEUTRAL":
        st.warning("The predicted sentiment is neutral")
    else:
        st.warning("The predicted sentiment is negative")
    st.success(result)
