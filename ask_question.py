import sys

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

    vectorstore = FAISS.load_local(
        "vectorstore",
        OpenAIEmbeddings(),
    )

    retriever = VectorStoreRetriever(vectorstore=vectorstore)

    retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

    response = retrievalQA.invoke(query)

    # print(f"Question: {response['query']}")
    print(f"Answer: {response['result']}")
