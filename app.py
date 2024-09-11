import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar for Groq API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Input for YouTube or Website URL
generic_url = st.text_input("URL", label_visibility="collapsed")

# Validate if the Groq API key is correctly passed
if groq_api_key:
    st.write("Groq API Key successfully captured.")
else:
    st.error("Please enter a valid Groq API Key.")

# Define the language model using Groq API Key
if groq_api_key.strip():
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
else:
    st.stop()

# Define the prompt template for summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization process triggered by button click
if st.button("Summarize the Content from YT or Website"):
    
    # Validate URL and API key input
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                
                # Loading data from YouTube or website
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                                          "Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Displaying the result
                st.success(output_summary)
        
        except Exception as e:
            st.exception(f"Exception: {e}")
