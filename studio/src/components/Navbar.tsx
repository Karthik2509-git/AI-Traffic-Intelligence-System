import React from 'react';
import { Camera, Cpu, Zap, Wifi, Smartphone } from 'lucide-react';
import type { TelemetryData } from '../types';

interface NavbarProps {
  telemetry: TelemetryData;
  onOpenMobileCam: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ telemetry, onOpenMobileCam }) => {
  const [timeStr, setTimeStr] = React.useState<string>('');

  React.useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTimeStr(now.toTimeString().split(' ')[0]);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

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

        <div className="badge badge-cyan">
          C++ ENGINE v3.1
        </div>
        <div className="badge badge-green">
          CUDA 12.4 • TensorRT 10
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '24px', fontSize: '0.82rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-dim)' }}>
          <Wifi size={14} color="var(--accent-green)" />
          <span>FPS: <strong style={{ color: 'var(--accent-cyan)' }}>{telemetry.fps.toFixed(1)}</strong></span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-dim)' }}>
          <Zap size={14} color="var(--accent-orange)" />
          <span>Latency: <strong style={{ color: 'var(--accent-orange)' }}>{telemetry.latency_ms.toFixed(1)} ms</strong></span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-dim)' }}>
          <Camera size={14} color="var(--accent-cyan)" />
          <span>Active Nodes: <strong style={{ color: '#fff' }}>{telemetry.active_cameras}</strong></span>
        </div>

        <button 
          onClick={onOpenMobileCam}
          className="btn-primary"
          style={{ fontSize: '0.8rem', padding: '6px 12px' }}
        >
          <Smartphone size={14} />
          Connect Mobile Cam
        </button>

        <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          {timeStr || '00:00:00'}
        </div>
      </div>
    </header>
  );
};
