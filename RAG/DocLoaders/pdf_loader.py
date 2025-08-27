from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('<filePath>')

docs = loader.load()

