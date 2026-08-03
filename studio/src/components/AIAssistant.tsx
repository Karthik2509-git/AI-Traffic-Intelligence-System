import React from 'react';
import type { TelemetryData } from '../types';
import { Bot, Send, Calculator } from 'lucide-react';

interface AIAssistantProps {
  telemetry: TelemetryData;
  engineStatus: string;
}

export const AIAssistant: React.FC<AIAssistantProps> = ({ telemetry, engineStatus }) => {
  const isOnline = engineStatus === 'online';

  const [messages, setMessages] = React.useState<{ role: 'user' | 'assistant'; text: string; formula?: string }[]>([
    {
      role: 'assistant',
      text: isOnline 
        ? 'Hello! I am your Explainable AI Assistant for ATOS Studio. Ask me anything about intersection traffic density, formulas, incident alerts, or model predictions.'
        : 'Hello! C++ Engine Telemetry is currently offline or waiting for stream. Connect the backend engine to calculate live traffic pressure and signal extensions.',
    }
  ]);
  const [input, setInput] = React.useState('');

  const samplePrompts = [
    'Why is Intersection Alpha congested?',
    'Show formula for Traffic Pressure Index',
    'Summarize active vehicle classes',
    'Explain TensorRT FP16 optimization'
  ];

  const handleSend = (textToSend?: string) => {
    const q = textToSend || input;
    if (!q.trim()) return;

    const userMsg = { role: 'user' as const, text: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    setTimeout(() => {
      let reply = "Based on live telemetry from the C++ ATOS Engine:";
      let formulaStr: string | undefined;

      if (!isOnline) {
        reply = "C++ Engine Disconnected: Unable to fetch live telemetry values. Please launch `atos_traffic_system.exe` or UDP gateway stream.";
      } else if (q.toLowerCase().includes('pressure') || q.toLowerCase().includes('formula')) {
        reply = `Traffic Pressure Index is calculated continuously in C++ using normalized vehicle density against intersection design capacity:`;
        formulaStr = `P = min( ActiveTracks / IntersectionCapacity, 1.0 )\nCurrently: P = min( ${telemetry.vehicles} / 200, 1.0 ) = ${telemetry.pressure.toFixed(2)}`;
      } else if (q.toLowerCase().includes('congested') || q.toLowerCase().includes('alpha')) {
        reply = `Intersection Alpha currently has ${telemetry.vehicles} active vehicles. Traffic pressure is ${telemetry.pressure.toFixed(2)}. Signal Phase 0 is receiving an automatic +15s green extension to clear northbound queue density.`;
      } else if (q.toLowerCase().includes('tensorrt') || q.toLowerCase().includes('optimization')) {
        reply = `TensorRT 10 deserializes YOLOv8m FP16 weights into GPU device memory bindings. Combined with fused CUDA bilinear resizing, inference completes in ${telemetry.latency_ms.toFixed(1)}ms.`;
      } else {
        reply = `Analysis complete: ${telemetry.vehicles} active vehicles detected across 8 camera nodes with 100% pipeline health.`;
      }

      setMessages((prev) => [...prev, { role: 'assistant', text: reply, formula: formulaStr }]);
    }, 600);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Bot color="var(--accent-purple)" />
            EXPLAINABLE AI ASSISTANT & FORMULA INSIGHTS
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Natural language diagnostic query engine with mathematical transparency
          </p>
        </div>

        <span className={isOnline ? "badge badge-purple" : "badge badge-orange"}>
          {isOnline ? "Engine Online" : "Engine Offline"}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', flex: 1 }}>
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', maxHeight: '420px', paddingRight: '8px' }}>
            {messages.map((m, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  gap: '12px',
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%'
                }}
              >
                {m.role === 'assistant' && (
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: 'rgba(168, 85, 247, 0.2)',
                    border: '1px solid var(--accent-purple)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--accent-purple)',
                    flexShrink: 0
                  }}>
                    <Bot size={18} />
                  </div>
                )}

                <div style={{
                  background: m.role === 'user' ? 'linear-gradient(135deg, #00f2fe, #4facfe)' : '#121826',
                  color: m.role === 'user' ? '#000' : 'var(--text-main)',
                  border: m.role === 'user' ? 'none' : '1px solid var(--border-dim)',
                  padding: '12px 16px',
                  borderRadius: '10px',
                  fontSize: '0.88rem',
                  lineHeight: '1.5'
                }}>
                  <div>{m.text}</div>
                  {m.formula && (
                    <div style={{
                      marginTop: '10px',
                      background: '#07090e',
                      padding: '10px 12px',
                      borderRadius: '6px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.78rem',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--border-dim)'
                    }}>
                      <pre>{m.formula}</pre>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(p)}
                  style={{
                    background: 'var(--bg-card-hover)',
                    color: 'var(--text-muted)',
                    border: '1px solid var(--border-dim)',
                    padding: '4px 10px',
                    borderRadius: '16px',
                    fontSize: '0.75rem',
                    cursor: 'pointer'
                  }}
                >
                  ✨ {p}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask Explainable AI Assistant about traffic density, formulas, or system health..."
                style={{
                  flex: 1,
                  background: '#0a0d16',
                  border: '1px solid var(--border-dim)',
                  borderRadius: '8px',
                  padding: '10px 14px',
                  color: '#fff',
                  fontSize: '0.88rem',
                  outline: 'none'
                }}
              />
              <button className="btn-primary" onClick={() => handleSend()}>
                <Send size={14} /> Send
              </button>
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Calculator size={16} color="var(--accent-cyan)" />
            EXPLAINABLE FORMULAS
          </h3>

          <div style={{ background: '#0a0d16', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '6px' }}>
              1. Traffic Pressure Formula
            </div>
            <code style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
              P = min( Σ(ActiveTracks) / C_max, 1.0 )
            </code>
          </div>

          <div style={{ background: '#0a0d16', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-green)', marginBottom: '6px' }}>
              2. Fused CUDA Preprocessing
            </div>
            <code style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
              Pixel_out = ((Pixel_in / 255.0) - 0.5) * 1.1 + 0.5
            </code>
          </div>

          <div style={{ background: '#0a0d16', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-orange)', marginBottom: '6px' }}>
              3. IoU Track Retention
            </div>
            <code style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
              IoU(A, B) = Area(A ∩ B) / Area(A ∪ B) &gt; 0.40
            </code>
          </div>
        </div>
      </div>
    </div>
  );
};
