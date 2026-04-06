# app/chain.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from app.retriever import hybrid_retrieve
from app.config import get_settings

settings = get_settings()


#prompts
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a careful document analyst.

INSTRUCTIONS:
- Answer ONLY from the provided context
- Read ALL chunks completely before answering
- For location questions: schools before college = hometown not current address
- If answer not in context: say exactly "I don't have enough information to answer this"
- Think step by step for complex questions
- Always be concise and specific"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Context:
{context}

Question: {question}""")
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up question as a standalone question using chat history. Return ONLY the rewritten question, nothing else."),
    ("human", """Chat history:
{chat_history}

Follow-up: {question}

Standalone question:""")
])



def get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature
    )

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join([
        f"[Chunk {i+1} | Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

#chain builder

def build_rag_chain(vectorstore: FAISS, bm25: BM25Retriever) -> Runnable:
    """
    Full memory-aware RAG chain with query rewriting.
    Input:  {"question": str, "chat_history": list}
    Output: str
    """

    llm = get_llm()
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    def run(inputs: dict) -> str:
        question = inputs["question"]
        history  = inputs.get("chat_history", [])

        # Rewrite if there's history
        if history:
            history_text = "\n".join([
                f"{'User' if i%2==0 else 'Bot'}: {msg.content}"
                for i, msg in enumerate(history)
            ])
            standalone = rewrite_chain.invoke({
                "question":     question,
                "chat_history": history_text
            })
        else:
            standalone = question

        # Retrieve with question
        docs, _ = hybrid_retrieve(standalone, vectorstore, bm25)
        context  = format_docs(docs)

        # Generate with original question + history
        return (RAG_PROMPT | llm | StrOutputParser()).invoke({
            "context":      context,
            "question":     question,
            "chat_history": history
        })

    return RunnableLambda(run)

#streaming

def build_streaming_chain(vectorstore: FAISS, bm25: BM25Retriever) -> Runnable:
    """Same as RAG chain but LLM streams tokens"""

    streaming_llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        streaming=True
    )

    retriever_runnable = RunnableLambda(
        lambda q: format_docs(hybrid_retrieve(q, vectorstore, bm25)[0])
    )

    return (
        {
            "context":      retriever_runnable,
            "question":     RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: [])
        }
        | RAG_PROMPT
        | streaming_llm
        | StrOutputParser()
    )