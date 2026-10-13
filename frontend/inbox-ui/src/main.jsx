import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, Bot, BookOpen, Check, ChevronDown, CircleHelp, Clock3, Command,
  Filter, Globe, Inbox, Link2, MessageCircle, MoreHorizontal, Paperclip,
  Phone, Plus, Search, Send, Settings, ShieldCheck, Smile, Star, Tag,
  Users, Workflow, X, BarChart3, Bell, Sparkles, Instagram, MessageSquare,
  PanelLeft, CheckCircle2, Zap, CalendarDays, UserRound, FileText, Circle
} from 'lucide-react';
import './styles.css';

const navItems = [
  { label: 'Inbox', icon: Inbox, count: '12' },
  { label: 'All conversations', icon: MessageCircle },
  { label: 'AI agents', icon: Bot },
  { label: 'Knowledge base', icon: BookOpen },
  { label: 'Workflows', icon: Workflow },
  { label: 'Analytics', icon: BarChart3 },
  { label: 'Customers', icon: Users },
  { label: 'Integrations', icon: Link2 },
  { label: 'Settings', icon: Settings }
];

function channelLabel(ch) {
  return { whatsapp: 'WhatsApp', messenger: 'Messenger', web: 'Web chat', instagram: 'Instagram', telegram: 'Telegram' }[ch] || ch || 'Web chat';
}

function ChannelIcon({ channel }) {
  if (channel === 'whatsapp') return <span className="channel whatsapp">◉</span>;
  if (channel === 'instagram') return <Instagram size={17} className="instagram" />;
  if (channel === 'telegram') return <Send size={17} className="telegram" />;
  if (channel === 'messenger') return <MessageCircle size={17} className="telegram" />;
  return <Globe size={17} className="channel-web" />;
}

function Avatar({ initials, color, large = false }) {
  return <div className={`avatar ${color || 'blue'} ${large ? 'avatar-lg' : ''}`}>{initials}</div>;
}

function App() {
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [activeTab, setActiveTab] = useState('Open');
  const [search, setSearch] = useState('');
  const [message, setMessage] = useState('');
  const [replyMode, setReplyMode] = useState('Reply');
  const [section, setSection] = useState('Inbox');
  const [toggles, setToggles] = useState({ 'Auto-reply': true, 'Auto-resolve': true, 'Human handoff': true });
  const [note, setNote] = useState('');
  const [savedNote, setSavedNote] = useState('');
  const [filterOpen, setFilterOpen] = useState(false);
  const [aiState, setAiState] = useState('idle');
  const [aiInfo, setAiInfo] = useState(null);

  const refreshList = () => fetch('/api/inbox').then(r => r.json()).then(setChats);
  const refreshDetail = (id) => fetch(`/api/inbox/${id}`).then(r => r.json()).then(setDetail);

  useEffect(() => { refreshList(); }, []);
  useEffect(() => { if (chats.length && !activeId) setActiveId(chats[0].id); }, [chats, activeId]);
  useEffect(() => { if (activeId) { setDetail(null); refreshDetail(activeId); } }, [activeId]);

  const activeChat = chats.find(c => c.id === activeId);
  const messages = detail?.messages || [];
  const customer = detail?.customer || {};
  const history = chats.filter(c => c.customer_id === customer.id && c.id !== activeId);

  const filteredChats = useMemo(() => chats.filter(c => {
    const matchesSearch = `${c.name} ${c.preview} ${c.subject}`.toLowerCase().includes(search.toLowerCase());
    const matchesTab = activeTab === 'All' || c.status === activeTab;
    return matchesSearch && matchesTab;
  }), [chats, search, activeTab]);

  function sendMessage() {
    if (!message.trim() || !activeId) return;
    const text = message.trim();
    setMessage('');
    fetch(`/api/inbox/${activeId}/reply`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    }).then(r => r.json()).then(d => {
      if (d.conversation) setDetail(d.conversation);
      refreshList();
    });
  }

  function aiReply() {
    if (!activeId || aiState === 'working') return;
    setAiState('working'); setAiInfo(null);
    fetch(`/api/inbox/${activeId}/ai-reply`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    }).then(r => r.json()).then(d => {
      if (d.error) { setAiState('error'); setAiInfo(d.error); }
      else {
        setAiState('done');
        const bits = [];
        bits.push(d.verification?.passed ? 'verified ✅' : 'needs review ⚠️');
        if (d.citations?.length) bits.push(`${d.citations.length} citation(s)`);
        if (d.needs_approval) bits.push('needs human approval');
        setAiInfo(bits.join(' · '));
        setDetail(d.conversation); refreshList();
      }
    }).catch(() => { setAiState('error'); setAiInfo('request failed'); });
  }

  function closeConversation() {
    if (!activeId) return;
    fetch(`/api/inbox/${activeId}/close`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    }).then(() => { refreshList(); refreshDetail(activeId); setActiveTab('Resolved'); });
  }

  const counts = useMemo(() => ({
    Open: chats.filter(c => c.status === 'Open').length,
    Pending: chats.filter(c => c.status === 'Pending').length,
    Resolved: chats.filter(c => c.status === 'Resolved').length
  }), [chats]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => { setSection('Inbox'); setActiveTab('Open'); }} aria-label="Aion home">
          <span className="brand-mark"><span /></span><span>Aion</span>
        </button>
        <div className="global-search"><Search size={19}/><input placeholder="Search conversations, customers, or keywords..." /></div>
        <div className="top-actions">
          <button className="agent-online"><span className="green-dot"/> AI Agent Online <ChevronDown size={15}/></button>
          <button className="icon-button notification" aria-label="Notifications"><Bell size={20}/><i/></button>
          <button className="admin-button"><Avatar initials="AH" color="tan"/><span><b>Ahmed</b><small>Admin</small></span><ChevronDown size={15}/></button>
        </div>
      </header>

      <div className="workspace">
        <aside className="left-sidebar">
          <nav className="main-nav">
            {navItems.map(({label, icon: Icon, count}) => (
              <button key={label} className={`nav-item ${section === label ? 'selected' : ''}`} onClick={() => setSection(label)}>
                <Icon size={19}/><span>{label}</span>{count && <b className="nav-count">{count}</b>}
              </button>
            ))}
          </nav>
          <div className="agent-status-card">
            <div className="status-heading"><b>AI Agent Status</b><Activity size={18}/></div>
            <div className="online-pill"><span className="green-dot"/> Online</div>
            {Object.entries(toggles).map(([label, enabled]) => (
              <div className="toggle-row" key={label}>
                <span>{label === 'Auto-reply' ? <Sparkles size={15}/> : label === 'Auto-resolve' ? <ShieldCheck size={15}/> : <Users size={15}/>} {label}</span>
                <button className={`switch ${enabled ? 'on' : ''}`} onClick={() => setToggles(t => ({...t, [label]: !t[label]}))} aria-label={`Toggle ${label}`}><i/></button>
              </div>
            ))}
          </div>
        </aside>

        <section className="conversation-list">
          <div className="list-tabs">
            {['Open', 'Pending', 'Resolved'].map(tab => <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}<span>{counts[tab]}</span></button>)}
          </div>
          <div className="list-search-row">
            <div className="list-search"><Search size={17}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search conversations..."/>{search && <button onClick={() => setSearch('')}><X size={14}/></button>}</div>
            <button className={`filter-button ${filterOpen ? 'active-filter' : ''}`} onClick={() => setFilterOpen(v => !v)} aria-label="Filter conversations"><Filter size={17}/></button>
          </div>
          {filterOpen && <div className="filter-popover"><b>Filter conversations</b><p>Showing {activeTab.toLowerCase()} conversations</p><button onClick={() => {setActiveTab('All'); setFilterOpen(false);}}>Show all conversations</button></div>}
          <div className="chat-list">
            {filteredChats.map(chat => (
              <button key={chat.id} className={`chat-row ${activeId === chat.id ? 'current' : ''}`} onClick={() => setActiveId(chat.id)}>
                <Avatar initials={chat.initials} color={chat.color}/>
                <span className="chat-summary"><b>{chat.name}</b><small>{chat.subject || chat.preview}</small></span>
                <span className="chat-meta"><small>{chat.time}</small><span className="channel-wrap"><ChannelIcon channel={chat.channel}/>{chat.unread > 0 && <i>{chat.unread}</i>}</span></span>
              </button>
            ))}
            {filteredChats.length === 0 && <div className="empty-state">No conversations found.</div>}
          </div>
        </section>

        <main className="chat-panel">
          {!detail ? <div className="empty-state" style={{marginTop: 60}}>Select a conversation…</div> : <>
          <div className="chat-header">
            <Avatar initials={customer.initials} color={customer.color} large/>
            <div className="chat-title"><b>{customer.name}</b><small>{detail.subject} · {channelLabel(detail.channel)}</small></div>
            <span className="customer-chip">{customer.segment || 'Customer'}</span>
            {customer.churn_level && <span className="customer-chip">churn: {customer.churn_level}</span>}
            <button className="icon-button header-icon" aria-label="Star conversation"><Star size={18}/></button>
            <button className="icon-button header-icon" aria-label="More options"><MoreHorizontal size={19}/></button>
            <button className="close-button" onClick={closeConversation}><Check size={16}/> Close</button>
          </div>
          <div className="messages">
            <div className="date-divider"><span>Conversation</span></div>
            {messages.map((msg, i) => msg.role === 'customer' ? (
              <div className="message-group customer-message" key={i}>
                <div className="bubble customer-bubble">{msg.text}</div><small className="message-time">{msg.time}</small>
              </div>
            ) : (
              <div className="message-group agent-message" key={i}>
                <div className="agent-label"><span className="mini-aion"><span/></span><b>Aion</b><span className="ai-chip">AI agent</span></div>
                <div className="bubble agent-bubble">{msg.text}</div>
                <small className="message-time">{msg.time}</small>
              </div>
            ))}
          </div>
          <div className="composer">
            <div className="composer-tabs"><button className={replyMode === 'Reply' ? 'active' : ''} onClick={() => setReplyMode('Reply')}><MessageSquare size={15}/> Reply</button><button className={replyMode === 'Internal note' ? 'active' : ''} onClick={() => setReplyMode('Internal note')}>Internal note</button></div>
            <textarea value={message} onChange={e => setMessage(e.target.value)} onKeyDown={e => {if (e.key === 'Enter' && !e.shiftKey) {e.preventDefault(); sendMessage();}}} placeholder={replyMode === 'Reply' ? 'Type a message as the agent...' : 'Write an internal note...'} />
            {aiInfo && <div style={{padding: '6px 16px', fontSize: 11, color: aiState === 'error' ? '#c00' : '#07813d'}}>{aiState === 'error' ? `AI error: ${aiInfo}` : `AI reply — ${aiInfo}`}</div>}
            <div className="composer-tools">
              <div className="tool-icons"><button title="Attach file"><Paperclip size={19}/></button><button title="Emoji"><Smile size={19}/></button><button title="Formatting"><span className="format-icon">T</span></button><button title="Quick actions"><Zap size={18}/></button><button title="Insert link"><Link2 size={18}/></button></div>
              <button className="assign-button" onClick={aiReply} disabled={aiState === 'working'} title="Generate a reply with the AI service agent"><Sparkles size={17}/> {aiState === 'working' ? 'Thinking…' : 'AI Reply'}</button>
              <button className="send-button" onClick={sendMessage}>Send <Send size={16}/><ChevronDown size={14}/></button>
            </div>
          </div>
          </>}
        </main>

        <aside className="customer-sidebar">
          {!detail ? null : <>
          <section className="customer-card profile-card">
            <div className="profile-top"><Avatar initials={customer.initials} color={customer.color} large/><div className="profile-name"><b>{customer.name}</b><small>{customer.email}</small></div><button className="icon-button"><MoreHorizontal size={19}/></button></div>
            <div className="profile-details">
              <div><UserRound/><span>Customer ID</span><b>{customer.id}</b></div>
              <div><FileText/><span>Segment</span><b>{customer.segment || '—'}</b></div>
              <div><Circle/><span>Churn risk</span><b>{customer.churn_score != null ? `${customer.churn_score} (${customer.churn_level})` : '—'}</b></div>
              <div><MessageCircle/><span>Language</span><b>Arabic</b></div>
              <div><Star/><span>Lifetime value</span><b>{customer.lifetime_value != null ? `${customer.lifetime_value.toLocaleString()} EGP` : '—'}</b></div>
            </div>
          </section>
          <section className="customer-card contact-card">
            <div><MessageSquare/><span>Channel</span><b>{channelLabel(detail.channel)}</b></div>
            <div><Phone/><span>Phone</span><b dir="ltr">{customer.phone || '—'}</b></div>
          </section>
          <section className="customer-card tags-card"><h3>Tags</h3><div className="tags">{(customer.tags || []).map(t => <span key={t} className="tag billing-tag"><Tag size={13}/> {t}</span>)}<button className="add-tag"><Plus size={16}/></button></div></section>
          <section className="customer-card notes-card"><div className="section-title"><h3>Notes</h3><button className="icon-button"><MoreHorizontal size={17}/></button></div><textarea value={note} onChange={e => setNote(e.target.value)} placeholder="Write a note..."/><div className="note-actions"><button onClick={() => {setSavedNote(note); setNote('');}}>Save note</button>{savedNote && <span>Saved</span>}</div>{savedNote && <p className="saved-note">{savedNote}</p>}</section>
          <section className="customer-card history-card"><h3>Other tickets</h3>
            {history.length === 0 && <p style={{fontSize: 11, color: '#7a8799'}}>No other tickets.</p>}
            {history.map(h => <div className="history-row" key={h.id}><span className="history-icon"><Check size={13}/></span><b>{h.subject}</b><span className="resolved-tag">{h.status}</span><small>{h.time}</small></div>)}
          </section>
          </>}
        </aside>
      </div>
      {section !== 'Inbox' && <div className="section-toast"><div><b>{section}</b><span>This UI demo is ready for this section to be connected.</span></div><button onClick={() => setSection('Inbox')}><X size={17}/></button></div>}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
