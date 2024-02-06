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

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display =  f"""<iframe
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 482px;">"""
    return pdf_display

def main():
    st.set_page_config(
    page_title="KIIT Chat Bot",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)
    from PIL import Image

    # Open the image file
    img = Image.open('logo.png')

    # Convert the image to base64
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Display the image in the center
    st.markdown(f'<p style="text-align: center;"><img src="data:image/png;base64,{img_str}" width="120"></p>', unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>KIIT Chat Bot</h1>", unsafe_allow_html=True)

    st.write("Ask questions regarding KIIT School of Computer Engineering Student Handbook")
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the PDF in the first column
    file = "KIIT_CSE_SHB_Updated-1.pdf"
    pdf_display = displayPDF(file)
    col1.markdown(pdf_display, unsafe_allow_html=True)

    # Display the Q&A in the second column
    user_question = col2.text_input("Ask a Question from the PDF Files")
    if user_question:
        response = user_input(user_question)  # Generate the response
        print(response)
        col2.write("Reply: " + response["output_text"])

if __name__ == "__main__":
    main()
