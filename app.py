import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
import re
from langchain_community.vectorstores import FAISS


load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as precise as possible from the provided context, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = OpenAIEmbeddings()

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    

    # Using CSV parser
    output_parser = CommaSeparatedListOutputParser()
    #format_instructions = output_parser.get_format_instructions()

    # prompt2 = PromptTemplate(
    #     template="List  {subject}.\n{format_instructions}",
    #     input_variables=["subject"],
    #     partial_variables={"format_instructions": format_instructions},
    # )
    #prompt_message = prompt2.format(subject=user_question)
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    if re.search(r"\blist\b", user_question, re.IGNORECASE):
        output_parser = CommaSeparatedListOutputParser()
        output = output_parser.parse(response["output_text"])
    else:
        output = response["output_text"]

    # output = output_parser.parse(response["output_text"])

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({"question": user_question, "response": output})

    st.write("### Conversation History:")
    for entry in st.session_state.history:
        st.write(f"**Question**: {entry['question']}")
        st.write(f"**Response**: {entry['response']}")
        st.write("---")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with multiple PDFs ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
