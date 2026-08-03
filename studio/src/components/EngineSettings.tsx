import React from 'react';
import { Settings, Cpu, Save, Sliders } from 'lucide-react';

export const EngineSettings: React.FC = () => {
  const [model, setModel] = React.useState('data/yolov8_4k_optimized.engine');
  const [precision, setPrecision] = React.useState<'FP16' | 'FP32' | 'INT8'>('FP16');
  const [conf, setConf] = React.useState(0.20);
  const [nms, setNms] = React.useState(0.55);
  const [saved, setSaved] = React.useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Settings color="var(--accent-cyan)" />
            ENGINE & MODEL SETTINGS
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Configure TensorRT optimization precision, YOLO model weights, and telemetry destinations
          </p>
        </div>

        <button className="btn-primary" onClick={handleSave}>
          <Save size={14} /> {saved ? 'Configuration Saved!' : 'Save & Reload Engine'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="var(--accent-cyan)" />
            TENSORRT ENGINE CONFIGURATION
          </h3>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Serialized Engine Path (.engine)
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                width: '100%',
                background: '#07090e',
                border: '1px solid var(--border-dim)',
                borderRadius: '6px',
                padding: '8px 12px',
                color: '#fff',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-mono)'
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Execution Precision
            </label>
            <div style={{ display: 'flex', gap: '10px' }}>
              {(['FP16', 'FP32', 'INT8'] as const).map((p) => (
                <button
                  key={p}
                  onClick={() => setPrecision(p)}
                  className={precision === p ? 'btn-primary' : 'btn-secondary'}
                  style={{ flex: 1, fontSize: '0.8rem' }}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={16} color="var(--accent-green)" />
            DETECTION & NMS THRESHOLDS
          </h3>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
              <span>Confidence Threshold</span>
              <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{conf.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
              <span>NMS Threshold</span>
              <strong style={{ color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>{nms.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={nms}
              onChange={(e) => setNms(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-green)' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
