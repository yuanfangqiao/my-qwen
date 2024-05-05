import os
from abc import ABC
from typing import Any, List, Optional, Dict, cast

import torch
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from modelscope import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ServiceContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import set_global_service_context
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from llama_index.agent import VectorIndexRetriever

# configs for LLM
llm_name = "Qwen/Qwen-1_8B-Chat"
llm_revision = "master"

# configs for embedding model
embedding_model = "damo/nlp_gte_sentence-embedding_chinese-small"

# file path for your custom knowledge base
knowledge_doc_file_dir = "/home/yuanfangqiao/code/my-qwen1.5/embedding/custom_data"
knowledge_doc_file_path = knowledge_doc_file_dir + "xianjiaoda.md"


# define our Embedding class to use models in Modelscope
class ModelScopeEmbeddings4LlamaIndex(BaseEmbedding, ABC):
    embed: Any = None
    model_id: str = "damo/nlp_gte_sentence-embedding_chinese-small"

    def __init__(
            self,
            model_id: str,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from modelscope.models import Model
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            self.embed = pipeline(Tasks.sentence_embedding, model=self.model_id)

        except ImportError as e:
            raise ValueError(
                "Could not import some python packages." "Please install it with `pip install modelscope`."
            ) from e

    def _get_query_embedding(self, query: str) -> List[float]:
        text = query.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        return self.embed(input=inputs)['text_embedding'][0]

    def _get_text_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        return self.embed(input=inputs)['text_embedding'][0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        inputs = {"source_sentence": texts}
        return self.embed(input=inputs)['text_embedding']

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


# define our Retriever with llama-index to co-operate with Langchain
# note that the 'LlamaIndexRetriever' defined in langchain-community.retrievers.llama_index.py
# is no longer compatible with llamaIndex code right now.
class LlamaIndexRetriever(BaseRetriever):
    index: Any
    """LlamaIndex index to query."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.legacy.indices.base import BaseIndex
            from llama_index.core.base.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseIndex, self.index)
        print('@@@ query=', query)

        response = index.as_query_engine().query(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            print('@@@@ source=', source_node)
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.get_text(), metadata=metadata)
            )
        return docs

def torch_gc():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    DEVICE = "cuda"
    DEVICE_ID = "0"
    CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
    a = torch.Tensor([1, 2])
    a = a.cuda()
    print(a)

    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# global resources used by QianWenChatLLM (this is not a good practice)
tokenizer = AutoTokenizer.from_pretrained(llm_name, revision=llm_revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(llm_name, revision=llm_revision, device_map="auto",
                                             trust_remote_code=True, fp16=True).eval()


# define QianWen LLM based on langchain's LLM to use models in Modelscope
class QianWenChatLLM(LLM):
    max_length = 10000
    temperature: float = 0.01
    top_p = 0.9

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self):
        return "ChatLLM"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs: Any,
    ) -> str:
        print(prompt)
        response, history = model.chat(tokenizer, prompt, history=None)
        torch_gc()
        return response


# STEP1: create LLM instance
qwllm = QianWenChatLLM()
print('STEP1: qianwen LLM created')

# STEP2: load knowledge file and initialize vector db by llamaIndex
print('STEP2: reading docs ...')
embeddings = ModelScopeEmbeddings4LlamaIndex(model_id=embedding_model)
service_context = ServiceContext.from_defaults(embed_model=embeddings, llm=None)
set_global_service_context(service_context)     # global config, not good

llamaIndex_docs = SimpleDirectoryReader(knowledge_doc_file_dir).load_data()
llamaIndex_index = GPTVectorStoreIndex.from_documents(llamaIndex_docs, chunk_size=512)
retriever = LlamaIndexRetriever(index=llamaIndex_index)
print(' 2.2 reading doc done, vec db created.')

# STEP3: create chat template
prompt_template = """请基于```内的内容回答问题。"
```
{context}
```
我的问题是：{question}。
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)
print('STEP3: chat prompt template created.')

# STEP4: create RAG chain to do QA
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | qwllm
        | StrOutputParser()
)
chain.invoke('西安交大的校训是什么？')
# chain.invoke('魔搭社区有哪些模型?')
# chain.invoke('modelscope是什么?')
# chain.invoke('萧峰和乔峰是什么关系?')