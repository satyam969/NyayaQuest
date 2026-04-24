from .firebase_config import db
from google.cloud import firestore
import uuid
import datetime

def get_user_conversations(user_id):
    """Fetches all conversation threads for a specific user, sorted by recency."""
    if not db: return []
    
    try:
        docs = db.collection("conversations") \
                 .where("user_id", "==", user_id) \
                 .stream()
        
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        return []

def create_conversation(user_id, initial_title="New Chat"):
    """Creates a new conversation record in Firestore."""
    if not db: return str(uuid.uuid4())
    
    conversation_id = str(uuid.uuid4())
    convo_ref = db.collection("conversations").document(conversation_id)
    
    convo_data = {
        "user_id": user_id,
        "title": initial_title,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP
    }
    
    convo_ref.set(convo_data)
    return conversation_id

def update_conversation_title(conversation_id, new_title):
    """Updates the title of a conversation thread."""
    if not db: return
    db.collection("conversations").document(conversation_id).update({
        "title": new_title,
        "updated_at": firestore.SERVER_TIMESTAMP
    })

def add_message_to_db(conversation_id, role, content, metadata=None):
    """Appends a message to the conversation's history sub-collection."""
    if not db: return
    
    history_ref = db.collection("conversations").document(conversation_id).collection("history")
    
    message_data = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    history_ref.add(message_data)
    
    # Update the parent conversation's timestamp
    db.collection("conversations").document(conversation_id).update({
        "updated_at": firestore.SERVER_TIMESTAMP
    })

def get_conversation_history(conversation_id):
    """Retrieves all messages for a specific conversation ID."""
    if not db: return []
    
    try:
        messages = db.collection("conversations").document(conversation_id) \
                     .collection("history") \
                     .order_by("timestamp", direction=firestore.Query.ASCENDING) \
                     .stream()
        
        return [doc.to_dict() for doc in messages]
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []
