import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Only check if extracted_text is not None or empty
                text += " " + extracted_text  # Add a space before appending the text
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response
    #print(response)
    #st.write("Reply: ", response["output_text"])


#def main():
#    st.set_page_config("Chat PDF")
#    st.header("Chat with PDF using Gemini")

#    user_question = st.text_input("Ask a Question from the PDF Files")

#    if user_question:
#        user_input(user_question)
#        pdf_docs = ["KIIT_CSE_SHB_Updated-1.pdf"]
#        raw_text = get_pdf_text(pdf_docs)
#        text_chunks = get_chunks(raw_text)
#        get_vector_store(text_chunks)

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the PDF in the first column
    def get_base64_of_pdf(pdf_path):
        with open(pdf_path, "rb") as file:
            encoded_pdf = base64.b64encode(file.read()).decode("utf-8")
        return encoded_pdf

    pdf_docs = ["KIIT_CSE_SHB_Updated-1.pdf"]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_chunks(raw_text)
    get_vector_store(text_chunks)
    b64_pdf = get_base64_of_pdf(pdf_docs[0])
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="500px" type="application/pdf"></iframe>'
    col1.markdown(pdf_display, unsafe_allow_html=True)

    # Display the Q&A in the second column
    user_question = col2.text_input("Ask a Question from the PDF Files")
    if user_question:
        response = user_input(user_question)  # Generate the response
        print(response)
        col2.write("Reply: " + response["output_text"])
        #col2.write(response)  # Display the response

if __name__ == "__main__":
    main()