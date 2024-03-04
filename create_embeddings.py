import requests
import xmltodict
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

sitemap = requests.get("https://techknowcrave.com/sitemap.xml")
sitemap_xml = sitemap.text
raw_data = xmltodict.parse(sitemap_xml)

pages = []

# Extract text from each url in the sitemap and store it in the pages list
for info in raw_data["urlset"]["url"]:
    url = info["loc"]
    my_url = "https://techknowcrave.com/post/"
    if my_url in url:
        html = requests.get(url).text
        text = BeautifulSoup(html, features="html.parser").get_text()

        lines = (line.strip() for line in text.splitlines())
        cleaned_text = "\n".join(line for line in lines if line)
        pages.append({"text": cleaned_text, "source": url})

# Split the text into chunks of 1500 characters and store the chunks in the docs list
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs, metadatas = [], []
for page in pages:
    splits = text_splitter.split_text(page["text"])
    docs.extend(splits)
    metadatas.extend([{"source": page["source"]}] * len(splits))

vectorstore = FAISS.from_texts(
    docs,
    OpenAIEmbeddings(),
    metadatas=metadatas,
)

vectorstore.save_local("vectorstore")
