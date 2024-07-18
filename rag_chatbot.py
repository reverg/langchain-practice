import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


@st.cache_resource
def get_retriever_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever


@st.cache_resource
def build_chain(_llm, _retriever, _prompt):
    chain = (
        {
            "context": _retriever,
            "question": RunnablePassthrough(),
        }
        | _prompt
        | _llm
        | StrOutputParser()
    )

    return chain


def serve(chain):
    if "logs" not in st.session_state.keys():
        st.session_state.logs = [
            {"role": "assistant", "content": "How may I help you?"}
        ]

    for log in st.session_state.logs:
        with st.chat_message(log["role"]):
            st.write(log["content"])

    if question := st.chat_input():
        st.chat_message("user").write(question)
        req_log = {"role": "user", "content": question}
        st.session_state.logs.append(req_log)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
                st.write(answer)
        res_log = {"role": "assistant", "content": answer}
        st.session_state.logs.append(res_log)


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    retriever = get_retriever_from_pdf("./docs.pdf")
    prompt = hub.pull("rlm/rag-prompt")
    chain = build_chain(llm, retriever, prompt)
    serve(chain)
