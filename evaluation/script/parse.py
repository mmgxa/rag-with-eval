from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils.id_gen import generate_id


def parse_dir(dir_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFDirectoryLoader(dir_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    texts = text_splitter.split_documents(docs)
    docs = [
        {
            "source": text.metadata["source"].split("/")[-1],
            "page": text.metadata["page"],
            "text": text.page_content,
            "id": generate_id(text.page_content),
        }
        for text in tqdm(texts)
    ]
    return docs


def parse_file(file_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    texts = text_splitter.split_documents(docs)
    docs = [
        {
            "source": text.metadata["source"].split("/")[-1],
            "page": text.metadata["page"],
            "text": text.page_content,
            "id": generate_id(text.page_content),
        }
        for text in tqdm(texts)
    ]
    return docs
