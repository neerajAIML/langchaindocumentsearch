import streamlit as st
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import pickle

os.environ["OPENAI_API_KEY"] = "929b964d-406e-472d-9c76-d99aaf0a5b42:iBVYe0TDeuSRkiV64x9wKuCvOB0rnDlE"
os.environ["OPENAI_API_BASE"] = "https://openai.prod.ai-gateway.quantumblack.com/ff74be40-4d7c-4d8e-9aca-f119f2536639/v1"

load_dotenv()

def my_function():
    # Add your function logic here
    st.write("Function called!")
    vector_list = []
    folder_path = '/home/ai/project/Langchain-Document-Chat/data'
    for root, dirs, filenames in os.walk(folder_path):
        text = ""
        for file in filenames:
            
            fileL = os.path.join(root, file)
            print(fileL)
            pdf_reader = PdfReader(fileL)
            
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings()
        print(chunks)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        vector_list.append(VectorStore)
        print(vector_list)
        store_name = "master.pickle"
        vector_store_path = os.path.join('vector_store', store_name)    
        with open(vector_store_path, 'wb') as file:
                pickle.dump(VectorStore, file)
                print('wow')   
    print("Success fully done")
            
    
    

def main():
    
    st.header("Private Chat Window - You can ask any query 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    # Button to trigger the function
    

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
    
        store_name = pdf.name[:-4]+".pickle"
        vector_store_path = os.path.join('vector_store', store_name)
    else:
        store_name = "master.pickle"
        vector_store_path = os.path.join('vector_store', store_name) 
        

    if os.path.exists(vector_store_path):  # Check if vector store file exists
        print("I ready i had")
        with open(vector_store_path, 'rb') as file:
            VectorStore = pickle.load(file)
            print(VectorStore)
            print(type(VectorStore))

    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(vector_store_path, 'wb') as file:
            pickle.dump(VectorStore, file)  # Save the created vector store


    # Display chat history
    chat_history = st.session_state.get("chat_history", [])

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = OpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            chat_history.append({"user": query, "bot": response})

        st.session_state.chat_history = chat_history

        # Display chat history
        for i, message in enumerate(chat_history):
            with st.chat_message("user" if i % 2 == 0 else "assistant"):
                st.markdown(f"**{message['user']}**")
                st.markdown(message['bot'])

        # Display the latest bot response
        # st.write("Bot Response:", response)


if __name__ == "__main__":
    
    # if st.button("You want to chat with uploaded files"):
    #     my_function()
    # if st.button("You want to upload files to chat"):
    #my_function()
    main()