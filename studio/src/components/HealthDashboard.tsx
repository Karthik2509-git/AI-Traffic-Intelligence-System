import React from 'react';
import type { EngineMetrics } from '../types';
import { Activity, Cpu, HardDrive, Zap, Server, ShieldCheck, RefreshCw } from 'lucide-react';

interface HealthDashboardProps {
  metrics: EngineMetrics;
  engineStatus: string;
  onRefresh: () => void;
}

export const HealthDashboard: React.FC<HealthDashboardProps> = ({ metrics, engineStatus, onRefresh }) => {
  const isOnline = engineStatus === 'online';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity color="var(--accent-green)" />
            SYSTEM & HARDWARE HEALTH DIAGNOSTICS
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Real-time telemetry for GPU VRAM, CPU utilization, TensorRT execution, and frame drop queues
          </p>
        </div>

        <button className="btn-secondary" onClick={onRefresh}>
          <RefreshCw size={14} /> Refresh Hardware Diagnostics
        </button>
      </div>

      {/* Primary Hardware Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>CPU UTILIZATION</span>
            <Cpu size={18} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
            {metrics.cpu_utilization_pct !== undefined ? `${metrics.cpu_utilization_pct.toFixed(1)}%` : '--'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Host Multi-Threading Cores
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>GPU VRAM ALLOCATION</span>
            <Zap size={18} color="var(--accent-green)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
            {isOnline ? `${(metrics.vram_used_mb / 1024).toFixed(1)} GB` : '0 GB'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Total Allocated: {metrics.vram_total_mb / 1024} GB (Pinned DMA)
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>QUEUE DEPTH</span>
            <Server size={18} color="var(--accent-orange)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>
            {metrics.queue_depth} <span style={{ fontSize: '1rem' }}>pkts</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Lock-Free ConcurrentQueue Depth
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>DROPPED FRAMES</span>
            <HardDrive size={18} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-purple)' }}>
            {metrics.dropped_frames}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Backpressure Drops (0% Target)
          </div>
        </div>
      </div>

      {/* Hardware Specs & CUDA Execution Box */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={16} color="var(--accent-cyan)" />
            CUDA & TENSORRT ENGINE STATUS
          </h3>

          <div style={{ background: '#0a0d16', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px' }}>PRIMARY ACCELERATOR</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', fontFamily: 'var(--font-mono)' }}>
              {metrics.cuda_device}
            </div>
          </div>

          <div style={{ background: '#0a0d16', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px' }}>TENSORRT RUNTIME VERSION</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>
              TensorRT {metrics.tensorrt_version} ({metrics.precision} Precision)
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="var(--accent-green)" />
            PIPELINE STAGE LATENCY
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>
                <span>Fused CUDA Preprocess</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>1.2 ms</span>
              </div>
              <div style={{ width: '100%', height: '6px', background: '#0a0d16', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: '15%', height: '100%', background: 'var(--accent-cyan)' }}></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>
                <span>TensorRT Inference</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>8.4 ms</span>
              </div>
              <div style={{ width: '100%', height: '6px', background: '#0a0d16', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: '65%', height: '100%', background: 'var(--accent-green)' }}></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>
                <span>ByteTrack Post-Processing</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>2.1 ms</span>
              </div>
              <div style={{ width: '100%', height: '6px', background: '#0a0d16', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: '20%', height: '100%', background: 'var(--accent-orange)' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
