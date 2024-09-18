#pip install langchain openai python-dotenv

#from langchain.document_loaders import TextLoader #deprecated
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings  #deprecated
#from langchain_community.embeddings import OpenAIEmbeddings #deprecated
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
#import os

# Load Environment variables from .env
load_dotenv()



def main():
    ##load_dotenv()
    documents = load_docs()
    splitted_documents = text_splitter(documents)
    embeddings = OpenAIEmbeddings()
    #vector = embeddings.embed_query('testing the embedding model')
    #print(vector)
    #print(len(vector))

    #only the first 5 to save tokens for this example
    doc_vectors = embeddings.embed_documents([t.page_content for t in splitted_documents[:2]])
    print(doc_vectors)

def load_docs():

    load_dotenv()
    loader = TextLoader('transcrever_YouTube_video_IA.txt', encoding='utf-8')
    documents = loader.load()
    #print(docs) #print the document object
    #print(len(documents)) # 1 - we've olny read one file/document into the loader

    return documents

def text_splitter(documents):
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100 )
    splitted_texts = textSplitter.split_documents(documents)
    #print(texts[0])
    #print(len(texts))
    return splitted_texts

if __name__=="__main__":
     main()


