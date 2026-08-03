import React from 'react';
import { Cpu, Zap, Wifi, Smartphone, Shield, Bell } from 'lucide-react';
import type { TelemetryData, UserRole } from '../types';

interface NavbarProps {
  telemetry: TelemetryData;
  engineStatus: string;
  userRole: UserRole;
  setUserRole: (role: UserRole) => void;
  onOpenMobileCam: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  telemetry,
  engineStatus,
  userRole,
  setUserRole,
  onOpenMobileCam
}) => {
  const [timeStr, setTimeStr] = React.useState<string>('');
  const [showNotifs, setShowNotifs] = React.useState<boolean>(false);
  const [notifications, setNotifications] = React.useState<{ id: string; title: string; timestamp: string }[]>([
    { id: 'n1', title: 'ATOS Gateway Server Synced', timestamp: '00:00:01' },
    { id: 'n2', title: 'TensorRT FP16 Weights Loaded', timestamp: '00:00:05' }
  ]);

  React.useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTimeStr(now.toTimeString().split(' ')[0]);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const fetchNotifs = async () => {
    try {
      const res = await fetch('/notifications');
      if (res.ok) {
        const data = await res.json();
        if (data.notifications) setNotifications(data.notifications);
      }
    } catch (e) {
      // Diagnostic fallback
    }
  };

  React.useEffect(() => {
    fetchNotifs();
    const interval = setInterval(fetchNotifs, 3000);
    return () => clearInterval(interval);
  }, []);

  const isOnline = engineStatus === 'online';

  return (
    <header style={{
      height: '56px',
      background: '#0a0d16',
      borderBottom: '1px solid var(--border-dim)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      zIndex: 50
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px',
            height: '28px',
            borderRadius: '6px',
            background: 'linear-gradient(135deg, #00f2fe, #4facfe)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#000',
            fontWeight: '800'
          }}>
            <Cpu size={18} />
          </div>
          <span style={{ fontWeight: '800', letterSpacing: '0.5px', fontSize: '1.05rem', color: '#fff' }}>
            ATOS <span style={{ color: 'var(--accent-cyan)' }}>STUDIO</span>
          </span>
        </div>

        <div className={isOnline ? "badge badge-green" : "badge badge-orange"} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isOnline ? 'var(--accent-green)' : 'var(--accent-orange)',
            boxShadow: isOnline ? '0 0 8px var(--accent-green)' : 'none'
          }}></span>
          {isOnline ? 'C++ ENGINE ONLINE' : 'WAITING FOR ENGINE'}
        </div>

        <div className="badge badge-cyan">
          CUDA 12.4 • TensorRT 10
        </div>
      </div>

      {/* Real Indicators & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px', fontSize: '0.82rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-dim)' }}>
          <Wifi size={14} color={isOnline ? "var(--accent-green)" : "var(--text-muted)"} />
          <span>FPS: <strong style={{ color: isOnline ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            {isOnline ? telemetry.fps.toFixed(1) : 'Waiting'}
          </strong></span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-dim)' }}>
          <Zap size={14} color={isOnline ? "var(--accent-orange)" : "var(--text-muted)"} />
          <span>Latency: <strong style={{ color: isOnline ? 'var(--accent-orange)' : 'var(--text-muted)' }}>
            {isOnline ? `${telemetry.latency_ms.toFixed(1)} ms` : '--'}
          </strong></span>
        </div>

        <button 
          onClick={onOpenMobileCam}
          className="btn-primary"
          style={{ fontSize: '0.8rem', padding: '6px 12px' }}
        >
          <Smartphone size={14} />
          Connect Mobile Cam
        </button>

        {/* Notifications Tray */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowNotifs(!showNotifs)}
            className="btn-secondary"
            style={{ padding: '6px 10px', position: 'relative' }}
          >
            <Bell size={16} />
            {notifications.length > 0 && (
              <span style={{
                position: 'absolute',
                top: '-4px',
                right: '-4px',
                width: '14px',
                height: '14px',
                borderRadius: '50%',
                background: 'var(--accent-cyan)',
                color: '#000',
                fontSize: '0.65rem',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {notifications.length}
              </span>
            )}
          </button>

          {showNotifs && (
            <div className="glass-panel" style={{
              position: 'absolute',
              top: '40px',
              right: 0,
              width: '320px',
              padding: '16px',
              zIndex: 100,
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
            }}>
              <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#fff', borderBottom: '1px solid var(--border-dim)', paddingBottom: '8px' }}>
                SYSTEM NOTIFICATIONS
              </div>
              {notifications.map((n) => (
                <div key={n.id} style={{ fontSize: '0.78rem', padding: '6px', borderBottom: '1px solid #1e2638' }}>
                  <div style={{ color: '#fff', fontWeight: 600 }}>{n.title}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>{n.timestamp}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* RBAC Role Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#111726', padding: '4px 10px', borderRadius: '6px', border: '1px solid var(--border-dim)' }}>
          <Shield size={14} color="var(--accent-cyan)" />
          <select
            value={userRole}
            onChange={(e) => setUserRole(e.target.value as UserRole)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#fff',
              fontSize: '0.78rem',
              fontWeight: 600,
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            <option value="Guest" style={{ background: '#0a0d16' }}>Role: Guest</option>
            <option value="Developer" style={{ background: '#0a0d16' }}>Role: Developer</option>
            <option value="Administrator" style={{ background: '#0a0d16' }}>Role: Admin</option>
          </select>
        </div>

        <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          {timeStr || '00:00:00'}
        </div>
      </div>
    </header>
  );
};
