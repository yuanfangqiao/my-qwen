from langchain_openai import ChatOpenAI

my_base_url = "http://localhost:8000/v1/"
my_model = "Qwen1.5-7B-Chat"

llm = ChatOpenAI(api_key="EMPTY", base_url=my_base_url,model=my_model)


# msg = llm.invoke("please tell me the result of 1+7=?")
# print(f"recv msg:{msg.content}")

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "当前角色扮演，你是一个能解决问题的助手。"),
    ("user", "{input}")
])


from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

msg = chain.invoke("请回答1+15等于多少？")
print(f"basic,recv msg:{msg}")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

docs = loader.load()


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)


from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})


from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])

