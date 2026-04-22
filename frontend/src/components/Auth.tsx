import React, { useState } from 'react';
import axios from 'axios';
import { Scale, Lock, Mail, Loader2 } from 'lucide-react';

interface AuthProps {
  onLogin: (user: { user_id: string; email: string }) => void;
}

export default function Auth({ onLogin }: AuthProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const endpoint = isLogin ? '/api/auth/signin' : '/api/auth/signup';
      const response = await axios.post(`${apiUrl}${endpoint}`, {
        email,
        password
      });

      if (response.data.success) {
        onLogin({
          user_id: response.data.user_id,
          email: response.data.email
        });
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      width: '100vw',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg-color)',
      backgroundImage: 'radial-gradient(circle at 50% 0%, rgba(212, 175, 55, 0.05), transparent 40%)'
    }}>
      <div className="glass-panel" style={{
        padding: '40px',
        borderRadius: '16px',
        width: '100%',
        maxWidth: '420px',
        boxShadow: '0 20px 40px rgba(0,0,0,0.4)',
        border: '1px solid var(--border-color)',
        borderTop: '1px solid rgba(212, 175, 55, 0.3)'
      }}>
        
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <Scale size={48} color="var(--accent-gold)" style={{ marginBottom: '16px' }} />
          <h1 style={{ fontSize: '28px', margin: '0 0 8px 0' }}>NyayaQuest</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
            Advanced RAG Legal Assistant
          </p>
        </div>

        {error && (
          <div style={{ 
            background: 'rgba(239, 68, 68, 0.1)', 
            borderLeft: '3px solid var(--danger)',
            padding: '12px', 
            borderRadius: '4px',
            marginBottom: '20px',
            fontSize: '14px',
            color: '#fca5a5'
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <div style={{ position: 'relative' }}>
              <Mail size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
              <input 
                type="email" 
                placeholder="Email Address" 
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                style={{
                  width: '100%',
                  padding: '12px 12px 12px 40px',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'white',
                  fontSize: '15px'
                }}
              />
            </div>
          </div>

          <div>
            <div style={{ position: 'relative' }}>
              <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
              <input 
                type="password" 
                placeholder="Password" 
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                style={{
                  width: '100%',
                  padding: '12px 12px 12px 40px',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'white',
                  fontSize: '15px'
                }}
              />
            </div>
          </div>

          <button 
            type="submit" 
            className="btn-primary" 
            disabled={loading}
            style={{ 
              padding: '14px', 
              fontSize: '16px',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '8px',
              marginTop: '10px'
            }}
          >
            {loading ? <Loader2 className="animate-spin" size={20} /> : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px', color: 'var(--text-muted)' }}>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            type="button" 
            onClick={() => setIsLogin(!isLogin)}
            style={{ 
              background: 'none', 
              border: 'none', 
              color: 'var(--accent-gold)', 
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </div>
      </div>
    </div>
  );
}
