import React from 'react';
import type { TelemetryData } from '../types';
import { TrafficCone } from 'lucide-react';

interface SignalControllerViewProps {
  telemetry: TelemetryData;
}

export const SignalControllerView: React.FC<SignalControllerViewProps> = ({}) => {
  const [activePhase, setActivePhase] = React.useState<number>(0);
  const [timer, setTimer] = React.useState<number>(18);
  const [mode, setMode] = React.useState<'HEURISTIC' | 'REINFORCEMENT_LEARNING' | 'MANUAL'>('HEURISTIC');

  React.useEffect(() => {
    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev <= 1) {
          setActivePhase((p) => (p + 1) % 4);
          return 30;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const phases = [
    { id: 0, name: 'Phase 1: North-South Main Thoroughfare', green: activePhase === 0, yellow: timer < 4 && activePhase === 0 },
    { id: 1, name: 'Phase 2: East-West Arterial Flow', green: activePhase === 1, yellow: timer < 4 && activePhase === 1 },
    { id: 2, name: 'Phase 3: Northbound Left-Turn Bay', green: activePhase === 2, yellow: timer < 4 && activePhase === 2 },
    { id: 3, name: 'Phase 4: Southbound Pedestrian & Transit Clearance', green: activePhase === 3, yellow: timer < 4 && activePhase === 3 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <TrafficCone color="var(--accent-orange)" />
            NEMA ADAPTIVE SIGNAL OPTIMIZER
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Density-threshold heuristic & dynamic queue clearance engine
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>CONTROL MODE:</span>
          {(['HEURISTIC', 'REINFORCEMENT_LEARNING', 'MANUAL'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={mode === m ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.78rem', padding: '6px 12px' }}
            >
              {m.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative', minHeight: '340px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-muted)', position: 'absolute', top: '16px', left: '16px' }}>
            INTERSECTION-ALPHA SIGNAL STATE
          </div>

          <div style={{
            width: '180px',
            height: '180px',
            borderRadius: '50%',
            background: '#080c14',
            border: '4px solid var(--border-bright)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: activePhase === 0 ? '0 0 30px rgba(0, 255, 157, 0.4)' : '0 0 30px rgba(255, 159, 67, 0.3)'
          }}>
            <div style={{ fontSize: '2.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
              {timer}s
            </div>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginTop: '4px' }}>
              PHASE {activePhase + 1} ACTIVE
            </div>
          </div>

          <div className="badge badge-orange" style={{ marginTop: '20px' }}>
            ⚡ +15s GREEN EXTENSION GRANTED (Pressure &gt; 0.20)
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
            NEMA PHASE MATRIX & TIMING
          </h3>

          {phases.map((p) => (
            <div
              key={p.id}
              style={{
                background: p.green ? '#0d1d18' : 'var(--bg-card-hover)',
                border: p.green ? '1px solid var(--accent-green)' : '1px solid var(--border-dim)',
                padding: '14px',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: p.green ? (p.yellow ? 'var(--accent-orange)' : 'var(--accent-green)') : 'var(--accent-red)',
                  boxShadow: p.green ? '0 0 10px var(--accent-green)' : 'none'
                }}></div>
                <span style={{ fontSize: '0.88rem', fontWeight: 600, color: p.green ? '#fff' : 'var(--text-muted)' }}>
                  {p.name}
                </span>
              </div>

              {p.green ? (
                <span className="badge badge-green">RUNNING ({timer}s)</span>
              ) : (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>HOLDING</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
