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
  ShieldCheck
} from 'lucide-react';
import type { ActiveTab } from '../types';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'grid', label: 'Live Video Grid', icon: <Grid size={18} />, badge: '1-32' },
    { id: 'analytics', label: 'Traffic Analytics', icon: <BarChart3 size={18} /> },
    { id: 'signal', label: 'Signal Controller', icon: <TrafficCone size={18} />, badge: 'AUTO' },
    { id: 'automation', label: 'Automation (n8n)', icon: <Zap size={18} /> },
    { id: 'twin', label: '3D Digital Twin', icon: <Box size={18} />, badge: '3D' },
    { id: 'plugins', label: 'Plugin Marketplace', icon: <Store size={18} /> },
    { id: 'assistant', label: 'Explainable AI', icon: <Bot size={18} />, badge: 'AI' },
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
      justifyContent: 'space-between'
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div style={{ padding: '0 12px 12px 12px', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '1px' }}>
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
                padding: '10px 14px',
                borderRadius: '8px',
                fontSize: '0.88rem',
                fontWeight: isActive ? 600 : 500,
                color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                borderLeft: isActive ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ color: isActive ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span style={{
                  fontSize: '0.7rem',
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
        gap: '10px'
      }}>
        <ShieldCheck size={18} color="var(--accent-green)" />
        <div style={{ fontSize: '0.75rem' }}>
          <div style={{ color: 'var(--text-main)', fontWeight: 600 }}>Edge Node Active</div>
          <div style={{ color: 'var(--text-muted)' }}>Zero Cloud Dependency</div>
        </div>
      </div>
    </aside>
  );
};
