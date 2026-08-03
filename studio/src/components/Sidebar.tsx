import React from 'react';
import { 
  Grid, 
  BarChart3, 
  TrafficCone, 
  Zap, 
  Box, 
  Store, 
  Bot, 
  Settings,
  Activity,
  History,
  Terminal,
  ShieldCheck
} from 'lucide-react';
import type { ActiveTab } from '../types';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  engineStatus: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, engineStatus }) => {
  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'grid', label: 'Live Video Grid', icon: <Grid size={18} />, badge: '1-32' },
    { id: 'analytics', label: 'Traffic Analytics', icon: <BarChart3 size={18} /> },
    { id: 'signal', label: 'Signal Controller', icon: <TrafficCone size={18} />, badge: 'AUTO' },
    { id: 'automation', label: 'Automation (Flow)', icon: <Zap size={18} /> },
    { id: 'twin', label: '3D Digital Twin', icon: <Box size={18} />, badge: '3D' },
    { id: 'plugins', label: 'Plugin Store', icon: <Store size={18} /> },
    { id: 'assistant', label: 'Explainable AI', icon: <Bot size={18} />, badge: 'AI' },
    { id: 'health', label: 'System Health', icon: <Activity size={18} />, badge: 'SYS' },
    { id: 'replay', label: 'Traffic Replay', icon: <History size={18} /> },
    { id: 'logs', label: 'System Logs', icon: <Terminal size={18} /> },
    { id: 'settings', label: 'Engine Settings', icon: <Settings size={18} /> },
  ];

  return (
    <aside style={{
      width: '240px',
      background: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-dim)',
      padding: '16px 12px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      overflowY: 'auto'
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ padding: '0 12px 10px 12px', fontSize: '0.72rem', fontWeight: 800, color: 'var(--text-muted)', letterSpacing: '1px' }}>
          PLATFORM MODULES
        </div>

        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <div
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '9px 12px',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: isActive ? 600 : 500,
                color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                borderLeft: isActive ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ color: isActive ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span style={{
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: isActive ? 'rgba(0,242,254,0.2)' : 'rgba(255,255,255,0.05)',
                  color: isActive ? 'var(--accent-cyan)' : 'var(--text-dim)',
                  fontFamily: 'var(--font-mono)'
                }}>
                  {item.badge}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div style={{
        background: '#0d111a',
        border: '1px solid var(--border-dim)',
        padding: '12px',
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginTop: '16px'
      }}>
        <ShieldCheck size={18} color={engineStatus === 'online' ? "var(--accent-green)" : "var(--accent-orange)"} />
        <div style={{ fontSize: '0.75rem' }}>
          <div style={{ color: 'var(--text-main)', fontWeight: 600 }}>
            {engineStatus === 'online' ? 'C++ Engine Synced' : 'Gateway Listening'}
          </div>
          <div style={{ color: 'var(--text-muted)' }}>Zero Cloud Dependency</div>
        </div>
      </div>
    </aside>
  );
};
