import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def getLLamaresponse(input_text,no_words,style):
    llm=CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})

    template="""
        write a blog for {style} job profile for a topic {input_text}
        within {no_words} words    
        """
    
    prompt=PromptTemplate(input_variables=["style","input_text","no_words"],
                          template=template)

    response=llm(prompt.format(style=style,input_text=input_text,no_words=no_words))
    print(response)
    return response
    
st.set_page_config(page_title="Query Me",
                   page_icon='👁️‍🗨',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Query Me 👁️‍🗨")

input_text=st.text_input("Ask me anything")

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of words')
with col2:
    style=st.selectbox('Domain',
                            ('Researchers','Data scientist','Common People'),index=0)
    
submit=st.button("Generate")

if submit:
    st.write(getLLamaresponse(input_text,no_words,style))