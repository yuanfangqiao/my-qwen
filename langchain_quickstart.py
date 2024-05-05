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