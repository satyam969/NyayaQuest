import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Auth from './components/Auth';
import './index.css';

export interface User {
  user_id: string;
  email: string;
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);

  // Check local storage for session on load
  useEffect(() => {
    const savedUser = localStorage.getItem('nyayaquest_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogin = (loggedInUser: User) => {
    setUser(loggedInUser);
    localStorage.setItem('nyayaquest_user', JSON.stringify(loggedInUser));
  };

  const handleLogout = () => {
    setUser(null);
    setActiveThreadId(null);
    localStorage.removeItem('nyayaquest_user');
  };

  if (!user) {
    return <Auth onLogin={handleLogin} />;
  }

  return (
    <div className="app-container">
      <Sidebar
        user={user}
        activeThreadId={activeThreadId}
        onSelectThread={setActiveThreadId}
        onLogout={handleLogout}
      />
      <ChatInterface
        user={user}
        threadId={activeThreadId}
      />
    </div>
  );
}

export default App;
