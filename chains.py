from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

def get_rag_chain(llm, vector_store, system_prompt, qa_prompt, hybrid_retriever=None):
    from langchain.retrievers import MultiQueryRetriever
    
    # Custom Multi-Query prompting to prioritize formal Statutes and Sections
    template = """You are a senior Indian legal researcher with deep knowledge of the Bharatiya Nyaya Sanhita (BNS) 2023.
    The database contains chunks prefixed with [LAW_CODE YEAR] [CHAPTER] Section NUMBER.
    
    Rewrite the user's question into 3 different versions:
    1. A formal Section lookup (e.g., "Section regarding penalty for murder in BNS")
    2. An exact statutory phrase (e.g., "Whoever commits murder shall be punished with death")
    3. A chapter-aware legal research query (e.g., "Chapter VI offences affecting human body punishment for intentional culpable homicide")
    
    Original question: {question}
    Generate only the 3 versions:"""
    from langchain.prompts import PromptTemplate
    mq_prompt = PromptTemplate(input_variables=["question"], template=template)

    # Use hybrid retriever if provided, otherwise fall back to vector-only
    if hybrid_retriever is not None:
        base_retriever = hybrid_retriever
    else:
        base_retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 20}
        )
    
    # Multi-Query with statutory prompt
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, prompt=mq_prompt
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, mq_retriever, contextualize_q_prompt)
    
    qa_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", qa_prompt),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain