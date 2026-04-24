from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_conversation_title(llm, first_query):
    """Generates a short, descriptive title for a conversation based on the first query."""
    prompt = ChatPromptTemplate.from_template(
        "Generate a 3-5 word title for a legal conversation starting with this question: '{query}'. "
        "Return ONLY the title text, no quotes or prefix."
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        title = chain.invoke({"query": first_query})
        return title.strip()
    except Exception:
        return "Legal Query"
