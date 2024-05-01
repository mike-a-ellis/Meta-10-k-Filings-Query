import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
openai_chat_model_4 = ChatOpenAI(model="gpt-4-turbo")
# Split documents into chunks
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

docs = PyMuPDFLoader("Meta10k.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks, 
    embedding_model, 
    location=":memory:",
    collection_name="Meta10k",
)

#client = QdrantClient(path="./data/embeddings") 
#db = Qdrant(client=client, collection_name="Meta10k", embeddings=embedding_model,)

#qdrant_retriever = db.as_retriever()
qdrant_retriever = qdrant_vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})

@cl.on_chat_start
def chat_start():
    
    RAG_PROMPT = """
    You are an expert financial analyst.  You will be provided CONTEXT excerpts from the META company 10K annual report.  Your job is to answer the QUERY as correctly as you can using the information provided by the CONTEXT and your skills as an expert financial analyst. IF the context provided does give you enough information to answer the question, respond "I do not know"

    CONTEXT:
    {context}

    QUERY:
    {question}
    """

    EVAL_SYSTEM_TEMPLATE = """You are an expert in analyzing the quality of a response.

    You should be hyper-critical.

    Provide scores (out of 10) for the following attributes:

    1. Clarity - how clear is the response
    2. Faithfulness - how related to the original query is the response and the provided context
    3. Correctness - was the response correct?

    Please take your time, and think through each item step-by-step, when you are done - please provide your response in the following format:

    """

    EVAL_USER_TEMPLATE = """Query: {input}
    Context: {context}
    Response: {response}"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", EVAL_SYSTEM_TEMPLATE),
        ("human", EVAL_USER_TEMPLATE)
    ])

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=qdrant_retriever, llm=openai_chat_model)

    chain = ({"context": itemgetter("question") | retriever_from_llm, "question": itemgetter("question")} | RunnablePassthrough.assign(context=itemgetter("context")) | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")})
    eval_chain = eval_prompt | openai_chat_model_4

    cl.user_session.set("chain", chain)
    cl.user_session.set("eval_chain",eval_chain)

@cl.on_message
async def on_message(message: cl.Message):
    
    chain = cl.user_session.get("chain")
    eval_chain = cl.user_session.get("eval_chain")

    response = chain.invoke({"question":message.content})

    context = "\n".join([context.page_content for context in response["context"]])
    eval_response = eval_chain.invoke({"input":message.content, "context":context, "response":response["response"].content})

    await cl.Message(response["response"].content).send()
    await cl.Message(eval_response.content).send()

