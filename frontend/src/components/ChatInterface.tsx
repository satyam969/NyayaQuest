import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User as UserIcon, BookOpen, ChevronDown, ChevronUp, Loader2, Scale } from 'lucide-react';

interface User {
  user_id: string;
  email: string;
}

interface ChatInterfaceProps {
  user: User;
  threadId: string | null;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  context?: any[];
}

export default function ChatInterface({ user, threadId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedContexts, setExpandedContexts] = useState<Record<number, boolean>>({});
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (threadId) {
      loadHistory();
    } else {
      setMessages([]);
    }
  }, [threadId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const loadHistory = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.get(`${apiUrl}/api/conversations/${user.user_id}/${threadId}`);
      setMessages(response.data.history || []);
    } catch (error) {
      console.error("Failed to load chat history", error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleContext = (index: number) => {
    setExpandedContexts(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !threadId || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.post(`${apiUrl}/api/chat`, {
        user_id: user.user_id,
        thread_id: threadId,
        message: userMessage
      });

      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: response.data.response,
          context: response.data.context 
        }
      ]);
    } catch (error) {
      console.error("Chat error", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "An error occurred while processing your legal query. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  if (!threadId) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        <div style={{ textAlign: 'center' }}>
          <Scale size={64} color="rgba(212, 175, 55, 0.2)" style={{ marginBottom: '20px' }} />
          <h2>Select or start a new consultation</h2>
          <p style={{ marginTop: '10px' }}>Your queries are powered by advanced structural RAG</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      
      {/* Header */}
      <div className="glass-panel" style={{ padding: '20px 30px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: '18px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--success)' }}></span>
          Consultation Active
        </h3>
      </div>

      {/* Messages Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '30px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {messages.map((msg, idx) => (
          <div key={idx} className="animate-fade-in" style={{
            display: 'flex',
            gap: '16px',
            maxWidth: '850px',
            margin: msg.role === 'user' ? '0 0 0 auto' : '0 auto 0 0',
            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
          }}>
            
            {/* Avatar */}
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: msg.role === 'user' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(212, 175, 55, 0.15)',
              border: msg.role === 'user' ? '1px solid var(--accent-blue)' : '1px solid var(--accent-gold)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0
            }}>
              {msg.role === 'user' ? <UserIcon size={20} color="var(--accent-blue)" /> : <Bot size={20} color="var(--accent-gold)" />}
            </div>

            {/* Bubble */}
            <div style={{
              background: msg.role === 'user' ? 'rgba(59, 130, 246, 0.1)' : 'var(--panel-bg)',
              border: '1px solid var(--border-color)',
              padding: '20px',
              borderRadius: '12px',
              borderTopRightRadius: msg.role === 'user' ? '0' : '12px',
              borderTopLeftRadius: msg.role === 'assistant' ? '0' : '12px',
            }}>
              <div className="prose">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>

              {/* Citations block for assistant */}
              {msg.role === 'assistant' && msg.context && msg.context.length > 0 && (
                <div style={{ marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '16px' }}>
                  <button 
                    onClick={() => toggleContext(idx)}
                    style={{
                      background: 'none', border: 'none', color: 'var(--text-muted)',
                      display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px',
                      cursor: 'pointer', fontFamily: 'var(--font-sans)'
                    }}
                  >
                    <BookOpen size={14} />
                    {expandedContexts[idx] ? 'Hide Statutory Sources' : `View ${msg.context.length} Verified Sources`}
                    {expandedContexts[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>

                  {expandedContexts[idx] && (
                    <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {msg.context.map((doc, cIdx) => (
                        <div key={cIdx} className="citation-block">
                          <div className="citation-header">
                            <Scale size={12} />
                            {doc.metadata?.law_code || 'Unknown Law'} — {doc.metadata?.chapter || 'Unknown Chapter'} — Section {doc.metadata?.section_number || 'Unknown'}
                          </div>
                          <div style={{ color: 'var(--text-muted)', fontSize: '0.95em', lineHeight: '1.5' }}>
                            {doc.page_content}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="animate-fade-in" style={{ display: 'flex', gap: '16px', maxWidth: '850px' }}>
            <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(212, 175, 55, 0.15)', border: '1px solid var(--accent-gold)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Bot size={20} color="var(--accent-gold)" />
            </div>
            <div style={{ padding: '20px', borderRadius: '12px', background: 'var(--panel-bg)', border: '1px solid var(--border-color)', borderTopLeftRadius: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Loader2 className="animate-spin" size={18} color="var(--accent-gold)" />
              <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Analyzing legal corpus...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="glass-panel" style={{ padding: '20px 30px', borderTop: '1px solid var(--border-color)' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px', maxWidth: '850px', margin: '0 auto' }}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Describe your legal scenario or ask a statutory question..."
            disabled={loading}
            style={{
              flex: 1,
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid var(--border-color)',
              padding: '16px 20px',
              borderRadius: '12px',
              color: 'white',
              fontSize: '15px',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)'
            }}
          />
          <button 
            type="submit" 
            className="btn-primary"
            disabled={loading || !input.trim()}
            style={{ padding: '0 24px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Send size={18} />
            <span>Send</span>
          </button>
        </form>
        <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '11px', color: 'var(--text-muted)' }}>
          Responses are generated by AI and do not constitute formal legal advice.
        </div>
      </div>
    </div>
  );
}
