from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st
st.title("English to Tamil Translator")
model_name="jbochi/madlad400-3b-mt"
model=T5ForConditionalGeneration.from_pretrained(model_name,device_map='auto')
tokenizer=T5Tokenizer.from_pretrained(model_name)
text="<2ta> "+st.text_input("Enter your text")
input_ids=tokenizer(text,return_tensors='pt').input_ids.to(model.device)
outputs=model.generate(input_ids=input_ids)
op=tokenizer.decode(outputs[0],skip_special_tokens=True)
st.write(op)
