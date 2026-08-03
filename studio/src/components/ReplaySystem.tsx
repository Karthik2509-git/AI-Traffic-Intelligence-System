import React from 'react';
import { History, Play, Pause, Film, Calendar } from 'lucide-react';

export const ReplaySystem: React.FC = () => {
  const [isPlaying, setIsPlaying] = React.useState<boolean>(false);
  const [progress] = React.useState<number>(35);
  const [speed, setSpeed] = React.useState<number>(1.0);

  const incidentMarkers = [
    { time: '09:41:15', label: 'Wrong-Way Vehicle Alert', pct: 20 },
    { time: '09:43:50', label: 'High Traffic Density Spike (>80%)', pct: 55 },
    { time: '09:46:10', label: 'Signal Phase Extension Triggered', pct: 80 }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <History color="var(--accent-cyan)" />
            HISTORICAL TRAFFIC REPLAY & EVENT INDEX
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Replay recorded video streams with synchronized bounding box telemetry and incident bookmarks
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-secondary">
            <Calendar size={14} /> Select Date Range
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{
        aspectRatio: '16/9',
        maxHeight: '400px',
        background: '#07090e',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden'
      }}>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <Film size={48} color="var(--border-bright)" style={{ marginBottom: '12px' }} />
          <h3 style={{ fontSize: '1.05rem', color: '#fff', marginBottom: '4px' }}>Replay Frame Buffer Synchronized</h3>
          <p style={{ fontSize: '0.82rem' }}>Intersection Alpha — Stream Archive (09:40:00 to 09:50:00)</p>
        </div>

        <div style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          background: 'rgba(10, 13, 22, 0.85)',
          padding: '6px 12px',
          borderRadius: '6px',
          border: '1px solid var(--border-dim)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.78rem',
          color: 'var(--accent-cyan)'
        }}>
          REPLAY TIME: 09:43:30 (1x SPEED)
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <button
              className="btn-primary"
              onClick={() => setIsPlaying(!isPlaying)}
              style={{ width: '40px', height: '40px', padding: 0, justifyContent: 'center' }}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            </button>

            <div style={{ display: 'flex', gap: '8px' }}>
              {[0.5, 1.0, 2.0, 4.0].map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={speed === s ? 'btn-primary' : 'btn-secondary'}
                  style={{ fontSize: '0.75rem', padding: '4px 8px' }}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            09:43:30 / 09:50:00
          </div>
        </div>

        <div style={{ position: 'relative', width: '100%', height: '8px', background: '#0a0d16', borderRadius: '4px', cursor: 'pointer' }}>
          <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #00f2fe, #4facfe)', borderRadius: '4px' }}></div>

          {incidentMarkers.map((m, idx) => (
            <div
              key={idx}
              title={`${m.time}: ${m.label}`}
              style={{
                position: 'absolute',
                top: '-4px',
                left: `${m.pct}%`,
                width: '4px',
                height: '16px',
                background: 'var(--accent-red)',
                borderRadius: '2px',
                boxShadow: '0 0 8px var(--accent-red)'
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
