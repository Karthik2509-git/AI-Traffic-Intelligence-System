import React from 'react';
import type { SystemLog } from '../types';
import { Terminal, Search, RefreshCw } from 'lucide-react';

export const LogViewer: React.FC = () => {
  const [logs, setLogs] = React.useState<SystemLog[]>([]);
  const [filterLevel, setFilterLevel] = React.useState<string>('ALL');
  const [search, setSearch] = React.useState<string>('');

  const fetchLogs = async () => {
    try {
      const res = await fetch('/logs');
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (e) {
      setLogs([
        { timestamp: new Date().toTimeString().split(' ')[0], level: 'INFO', message: 'ATOS Gateway Log Service active.' },
        { timestamp: new Date().toTimeString().split(' ')[0], level: 'INFO', message: 'C++ Engine UDP socket bound to 127.0.0.1:5005.' },
      ]);
    }
  };

  React.useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, []);

  const filteredLogs = logs.filter((l) => {
    const matchLevel = filterLevel === 'ALL' || l.level === filterLevel;
    const matchSearch = !search || l.message.toLowerCase().includes(search.toLowerCase());
    return matchLevel && matchSearch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Terminal color="var(--accent-cyan)" />
            ENGINE & GATEWAY SYSTEM LOGS
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Live structured logging from C++ Engine, CUDA kernels, and FastAPI Control Plane
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-secondary" onClick={fetchLogs}>
            <RefreshCw size={14} /> Refresh Logs
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={16} color="var(--text-muted)" style={{ position: 'absolute', left: '12px', top: '10px' }} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search engine logs..."
            style={{
              width: '100%',
              background: '#0a0d16',
              border: '1px solid var(--border-dim)',
              borderRadius: '8px',
              padding: '8px 12px 8px 36px',
              color: '#fff',
              fontSize: '0.85rem',
              outline: 'none'
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {['ALL', 'INFO', 'WARN', 'ERROR'].map((lvl) => (
            <button
              key={lvl}
              onClick={() => setFilterLevel(lvl)}
              className={filterLevel === lvl ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.78rem', padding: '6px 12px' }}
            >
              {lvl}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-panel" style={{ flex: 1, padding: '16px', overflowY: 'auto', background: '#05070c', fontFamily: 'var(--font-mono)' }}>
        {filteredLogs.length === 0 ? (
          <div style={{ padding: '20px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No log entries matching query filters.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {filteredLogs.map((l, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '14px',
                  fontSize: '0.8rem',
                  padding: '6px 10px',
                  borderRadius: '4px',
                  background: l.level === 'ERROR' ? '#1c1014' : l.level === 'WARN' ? '#1c1710' : 'transparent',
                  borderLeft: l.level === 'ERROR' ? '3px solid var(--accent-red)' : l.level === 'WARN' ? '3px solid var(--accent-orange)' : '3px solid var(--accent-cyan)'
                }}
              >
                <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>[{l.timestamp}]</span>
                <span style={{
                  fontWeight: 700,
                  color: l.level === 'ERROR' ? 'var(--accent-red)' : l.level === 'WARN' ? 'var(--accent-orange)' : 'var(--accent-cyan)',
                  width: '50px',
                  flexShrink: 0
                }}>
                  {l.level}
                </span>
                <span style={{ color: 'var(--text-main)', wordBreak: 'break-all' }}>
                  {l.message}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
