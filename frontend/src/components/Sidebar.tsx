import { useEffect, useState } from 'react';
import axios from 'axios';
import { Scale, MessageSquare, PlusCircle, LogOut } from 'lucide-react';

interface User {
  user_id: string;
  email: string;
}

interface SidebarProps {
  user: User;
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onLogout: () => void;
}

interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}

export default function Sidebar({ user, activeThreadId, onSelectThread, onLogout }: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);

  const loadConversations = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.get(`${apiUrl}/api/conversations/${user.user_id}`);
      setConversations(response.data.conversations || []);
    } catch (error) {
      console.error("Failed to load conversations", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConversations();
  }, [user.user_id]);

  const handleNewChat = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.post(`${apiUrl}/api/conversations`, {
        user_id: user.user_id
      });
      const newThreadId = response.data.thread_id;
      onSelectThread(newThreadId);
      // Refresh list
      loadConversations();
    } catch (error) {
      console.error("Failed to create new chat", error);
    }
  };

  return (
    <div className="glass-panel" style={{
      width: '280px',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      padding: '20px 0',
      flexShrink: 0
    }}>

      {/* Brand Header */}
      <div style={{ padding: '0 20px', marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Scale size={28} color="var(--accent-gold)" />
        <span style={{ fontSize: '20px', fontFamily: 'var(--font-serif)', fontWeight: 600 }}>NyayaQuest</span>
      </div>

      {/* New Chat Button */}
      <div style={{ padding: '0 20px', marginBottom: '20px' }}>
        <button
          className="btn-primary"
          onClick={handleNewChat}
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
        >
          <PlusCircle size={18} />
          New Consultation
        </button>
      </div>

      {/* Chat List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 10px' }}>
        <h4 style={{ padding: '0 10px', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>
          Recent Consultations
        </h4>

        {loading ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '14px' }}>Loading...</div>
        ) : conversations.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '14px', fontStyle: 'italic' }}>
            No recent consultations found.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {conversations.map(conv => (
              <button
                key={conv.id}
                onClick={() => onSelectThread(conv.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '12px 16px',
                  background: activeThreadId === conv.id ? 'rgba(212, 175, 55, 0.1)' : 'transparent',
                  border: 'none',
                  borderLeft: activeThreadId === conv.id ? '3px solid var(--accent-gold)' : '3px solid transparent',
                  borderRadius: '0 8px 8px 0',
                  color: activeThreadId === conv.id ? 'white' : 'var(--text-main)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.2s ease',
                  width: '100%'
                }}
              >
                <MessageSquare size={16} color={activeThreadId === conv.id ? 'var(--accent-gold)' : 'var(--text-muted)'} />
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '14px' }}>
                  {conv.title}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* User Footer */}
      <div style={{ padding: '20px 20px 0 20px', borderTop: '1px solid var(--border-color)', marginTop: 'auto' }}>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '12px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {user.email}
        </div>
        <button
          className="btn-secondary"
          onClick={onLogout}
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
        >
          <LogOut size={16} />
          Sign Out
        </button>
      </div>

    </div>
  );
}
