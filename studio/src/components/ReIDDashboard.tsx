import React from 'react';
import { Target, Network, AlertTriangle, ShieldCheck, Cpu, RefreshCw, Layers, Database } from 'lucide-react';

interface ReIDSummary {
  reid_enabled: boolean;
  model_loaded: boolean;
  model_path: string;
  status: string;
  active_global_tracks: number;
  total_matches_found: number;
  similarity_threshold: number;
  benchmark: {
    status: string;
    evaluated?: boolean;
    rank1?: number | null;
    rank5?: number | null;
    mAP?: number | null;
    false_match_rate?: number | null;
    false_non_match_rate?: number | null;
    inference_ms?: number | null;
    matching_ms?: number | null;
    vram_used_mb?: number | null;
    dataset_name?: string | null;
    hardware?: string | null;
    message?: string;
  };
}

interface MatchRecord {
  global_vehicle_id: string;
  source_camera_id: string;
  target_camera_id: string;
  target_local_id: number;
  similarity_score: number;
  timestamp: string;
}

export const ReIDDashboard: React.FC = () => {
  const [summary, setSummary] = React.useState<ReIDSummary | null>(null);
  const [matches, setMatches] = React.useState<MatchRecord[]>([]);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);

  const fetchReIDData = async () => {
    try {
      setIsLoading(true);
      const resStatus = await fetch('/reid/status');
      if (resStatus.ok) {
        const data = await resStatus.json();
        setSummary(data);
      }

      const resMatches = await fetch('/reid/matches');
      if (resMatches.ok) {
        const matchData = await resMatches.json();
        setMatches(matchData.matches || []);
      }
    } catch (e) {
      console.log('ReID fetch error:', e);
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    fetchReIDData();
    const interval = setInterval(fetchReIDData, 4000);
    return () => clearInterval(interval);
  }, []);

  const isModelReady = summary?.model_loaded && summary?.reid_enabled;
  const benchmark = summary?.benchmark;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-card)',
        border: '1px solid var(--border-dim)',
        padding: '16px 20px',
        borderRadius: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #00f2fe, #4facfe)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#000'
          }}>
            <Target size={20} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff' }}>
              ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Multi-camera vehicle correlation, spatiotemporal windowing & feature matching
            </p>
          </div>
        </div>

        <button className="btn-secondary" onClick={fetchReIDData}>
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} /> Refresh Status
        </button>
      </div>

      {/* Model Readiness Status Warning Banner */}
      {!isModelReady && (
        <div style={{
          background: 'rgba(255, 159, 67, 0.12)',
          border: '1px solid var(--accent-orange)',
          padding: '16px 20px',
          borderRadius: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          color: '#ffc078'
        }}>
          <AlertTriangle size={24} color="var(--accent-orange)" />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fff' }}>
              Subsystem Diagnostic Status: {summary?.status || 'Re-ID model unavailable — evaluation pending'}
            </div>
            <div style={{ fontSize: '0.78rem', marginTop: '2px' }}>
              Single-camera YOLOv8 + ByteTrack pipeline is operating normally. Re-ID module fallback is active until a trained TensorRT model is loaded.
            </div>
          </div>
        </div>
      )}

      {/* Overview Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>RE-ID MODULE STATE</span>
            <ShieldCheck size={16} color={isModelReady ? "var(--accent-green)" : "var(--accent-orange)"} />
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 800, color: isModelReady ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
            {isModelReady ? 'ACTIVE' : 'FALLBACK (SAFE)'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            {summary?.reid_enabled ? 'reid_enabled: true' : 'reid_enabled: false (default)'}
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>ACTIVE GLOBAL TRACKS</span>
            <Layers size={16} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#fff' }}>
            {summary?.active_global_tracks || 0}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            Cross-camera identity clusters
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>SIMILARITY THRESHOLD</span>
            <Cpu size={16} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#fff' }}>
            {summary?.similarity_threshold ? `${(summary.similarity_threshold * 100).toFixed(0)}%` : '75%'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            Cosine vector distance cutoff
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>EVALUATION DATASETS</span>
            <Database size={16} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#fff' }}>
            VeRi-776 / CityFlow
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            Market-1501 excluded (person)
          </div>
        </div>
      </div>

      {/* Empirical Benchmark Report Card */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={18} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 800, color: '#fff' }}>
              Empirical Re-ID Model Benchmark Results
            </h3>
          </div>
          <span className="badge badge-cyan" style={{ fontSize: '0.72rem' }}>
            {benchmark?.evaluated ? `EMPIRICAL RESULT (${benchmark.dataset_name})` : 'EVALUATION PENDING'}
          </span>
        </div>

        {benchmark?.evaluated ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
            <div style={{ background: '#070a10', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>RANK-1 ACCURACY</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-green)' }}>
                {((benchmark.rank1 || 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: '#070a10', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>RANK-5 ACCURACY</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-green)' }}>
                {((benchmark.rank5 || 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: '#070a10', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>mAP SCORE</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
                {((benchmark.mAP || 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: '#070a10', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>INFERENCE LATENCY</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#fff' }}>
                {benchmark.inference_ms} ms
              </div>
            </div>
            <div style={{ background: '#070a10', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>MATCHING LATENCY</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#fff' }}>
                {benchmark.matching_ms} ms
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            background: '#070a10',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid var(--border-dim)',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.82rem'
          }}>
            <Database size={28} color="var(--accent-orange)" style={{ marginBottom: '8px' }} />
            <div style={{ fontWeight: 700, color: '#fff' }}>Evaluation Pending</div>
            <div style={{ marginTop: '4px', maxWidth: '500px', margin: '4px auto 0' }}>
              {benchmark?.message || 'Run scripts/benchmark_reid.py on VeRi-776 or CityFlow-ReID dataset to measure empirical Rank-1, Rank-5, and mAP accuracy metrics.'}
            </div>
          </div>
        )}
      </div>

      {/* Cross-Camera Matches Table */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Network size={18} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 800, color: '#fff' }}>
              Live Cross-Camera Identity Matches
            </h3>
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Total Matches: {matches.length}
          </span>
        </div>

        {matches.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-dim)', color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>GLOBAL ID</th>
                <th style={{ padding: '8px 12px' }}>ORIGIN CAM</th>
                <th style={{ padding: '8px 12px' }}>TARGET CAM</th>
                <th style={{ padding: '8px 12px' }}>SIMILARITY</th>
                <th style={{ padding: '8px 12px' }}>TIMESTAMP</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #1a2233' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {m.global_vehicle_id}
                  </td>
                  <td style={{ padding: '10px 12px', color: '#fff' }}>{m.source_camera_id}</td>
                  <td style={{ padding: '10px 12px', color: '#fff' }}>{m.target_camera_id}</td>
                  <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--accent-green)' }}>
                    {(m.similarity_score * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {m.timestamp}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            No cross-camera vehicle matches recorded yet. (Single-camera ByteTrack pipeline active).
          </div>
        )}
      </div>
    </div>
  );
};
