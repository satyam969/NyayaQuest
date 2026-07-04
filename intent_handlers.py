"""
Canned response handlers for non-legal intents.
No LLM calls, no retrieval — instant responses.
"""

from intent_router import QueryIntent


GREETING_RESPONSE = {
    "answer": (
        "👋 Hello! I'm **NyayaQuest** — your assistant for Indian statutory law.\n\n"
        "I can help you understand:\n"
        "• **Bharatiya Nyaya Sanhita (BNS) 2023** — criminal law\n"
        "• **Code of Civil Procedure (CPC) 1908** — civil procedures\n"
        "• **Right to Information Act 2005** — RTI\n"
        "• **Consumer Protection Act 2019** — consumer rights\n"
        "• **Labour Codes** — employment law\n\n"
        "**Try asking:**\n"
        "• *What are the grounds for anticipatory bail?*\n"
        "• *What is the punishment under Section 103 BNS?*\n"
        "• *How do I file an RTI application?*\n"
        "• *What are my rights as a consumer?*"
    ),
    "citations": [],
    "confidence": 1.0,
    "intent": QueryIntent.GREETING.value,
}


META_RESPONSE = {
    "answer": (
        "I'm **NyayaQuest**, an AI-powered legal research assistant built on "
        "Retrieval-Augmented Generation (RAG) technology.\n\n"
        "**What I can do:**\n"
        "✅ Answer questions about Indian statutory law\n"
        "✅ Cite specific sections and provisions from actual statutes\n"
        "✅ Explain legal concepts and procedures\n"
        "✅ Handle follow-up questions in a conversation\n\n"
        "**What I can't do:**\n"
        "❌ Provide legal advice for your specific case (consult a lawyer)\n"
        "❌ Access case law or judicial precedents\n"
        "❌ Cover state-specific laws (only central statutes)\n"
        "❌ Guarantee accuracy — always verify with primary sources\n\n"
        "**Covered acts:** BNS 2023, CPC 1908, RTI 2005, "
        "Consumer Protection Act 2019, Labour Codes.\n\n"
        "Ask me anything about the covered laws!"
    ),
    "citations": [],
    "confidence": 1.0,
    "intent": QueryIntent.META_QUERY.value,
}


OUT_OF_SCOPE_RESPONSE = {
    "answer": (
        "I'm specialized in **Indian statutory law**. My coverage is limited to:\n\n"
        "• Bharatiya Nyaya Sanhita (BNS) 2023\n"
        "• Code of Civil Procedure (CPC) 1908\n"
        "• Right to Information Act 2005\n"
        "• Consumer Protection Act 2019\n"
        "• Labour Codes\n\n"
        "Your question appears to be outside this scope. "
        "For general knowledge questions, please use ChatGPT, Google, or a "
        "general-purpose assistant.\n\n"
        "If you *do* have a legal question about the covered acts, "
        "please rephrase and I'll be happy to help!"
    ),
    "citations": [],
    "confidence": 1.0,
    "intent": QueryIntent.OUT_OF_SCOPE.value,
}


AMBIGUOUS_RESPONSE = {
    "answer": (
        "I'd love to help, but I need a bit more context to answer accurately.\n\n"
        "**Instead of a vague question, try being specific:**\n\n"
        "❌ *\"help me\"*\n"
        "✅ *\"Help me understand my rights as a tenant\"*\n\n"
        "❌ *\"what about section\"*\n"
        "✅ *\"What is Section 103 of BNS?\"*\n\n"
        "❌ *\"legal advice please\"*\n"
        "✅ *\"How do I file a consumer complaint?\"*\n\n"
        "The more specific your question, the more accurate my answer will be. "
        "Which of the covered acts are you asking about?"
    ),
    "citations": [],
    "confidence": 1.0,
    "intent": QueryIntent.AMBIGUOUS.value,
}


def handle_non_legal_intent(intent: QueryIntent) -> dict:
    """
    Return canned response for non-legal intents.
    Zero LLM cost, zero latency, no hallucination risk.
    """
    handlers = {
        QueryIntent.GREETING: GREETING_RESPONSE,
        QueryIntent.META_QUERY: META_RESPONSE,
        QueryIntent.OUT_OF_SCOPE: OUT_OF_SCOPE_RESPONSE,
        QueryIntent.AMBIGUOUS: AMBIGUOUS_RESPONSE,
    }
    return handlers.get(intent, AMBIGUOUS_RESPONSE).copy()
