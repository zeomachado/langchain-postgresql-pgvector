#pip install langchain openai python-dotenv

#from langchain.document_loaders import TextLoader #deprecated
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings  #deprecated
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

def load_doc():

    load_dotenv()

    loader = TextLoader('transcrever_YouTube_video_IA.txt', encoding='utf-8')

    documents = loader.load()
    return documents

def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100 )
    texts = text_splitter.split_documents(documents)
    #print(texts[0])
    print(len(texts))

if __name__=="__main__":
    load_dotenv()
    documents = load_doc()
    text_splitter(documents)

    #print(docs) #print the document object
    #print(len(documents)) # 1 - we've olny read one file/document into the loader


