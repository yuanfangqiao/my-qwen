from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

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


from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()

tools = [search]


from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

print(f"default prompt:[{prompt}]")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

rsp = agent_executor.invoke({"input": "how can langsmith help with testing?"})
print(f"step 1, rsp:{rsp}")


rsp = agent_executor.invoke({"input": "My have a friend named Ben"})
print(f"step 2, rsp:{rsp}")


chat_history = [HumanMessage(content="My name is Tom"), AIMessage(content="ok.")]
rsp = agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Did you remember my name?"
})

print(f"step 3, rsp:{rsp}")
