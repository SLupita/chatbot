from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
# from PyPDF2 import PdfReader
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdfloader = PyPDFLoader(pdf)
        pdfpages = pdfloader.load_and_split()

        # pdf_reader = PdfReader(pdf)
        # for page in pdf_reader.pages:
        #     text = page.extraxt.text()
    return text

def get_chunks(mydocuments):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(mydocuments)
    return texts

def get_vector_store(text_chunk, api_key):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(text_chunk, embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    chat_llm = ChatOpenAI(temperature = 0)
    prompt = PromptTemplate(template=prompt_template, input_variable = ["context","question"])
    chain = load_qa_chain(chat_llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(question, api_key):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

# Streamlit UI
def main():
    # Show title and description.
    st.title("💬 Chatbot")
    st.write(
        "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
        "It uses LangChain with RAG and ChatGPT"
    )

    # Ask user for their OpenAI API key via `st.text_input`.
    # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
    # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
    api_key = st.text_input("OpenAI API Key", type="password")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your docs", accept_multiple_files=True, key="pdf_uploader")
        if st.button("submit & process", ley="process_button") and api_key:
            with st.spinner("processing.."):
                text = get_pdf_text(pdf_docs)
                chunks = get_chunks(text)
                get_vector_store(chunks, api_key)
                st.success("Done")

if __name__ == "_main_":
    main()