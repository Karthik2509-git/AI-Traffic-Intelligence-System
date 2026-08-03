import React from 'react';
import type { TelemetryData } from '../types';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell 
} from 'recharts';
import { Activity, Car, Clock, ShieldAlert, Download, FileText } from 'lucide-react';

interface TrafficAnalyticsProps {
  telemetry: TelemetryData;
  engineStatus: string;
}

export const TrafficAnalytics: React.FC<TrafficAnalyticsProps> = ({ telemetry, engineStatus }) => {
  const [historyData, setHistoryData] = React.useState<{ time: string; pressure: number; vehicles: number; fps: number }[]>([]);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/analytics');
      if (res.ok) {
        const data = await res.json();
        if (data.history && data.history.length > 0) {
          setHistoryData(data.history);
        }
      }
    } catch (e) {
      // Keep existing data buffer
    }
  };

  React.useEffect(() => {
    fetchAnalytics();
    const timer = setInterval(fetchAnalytics, 1000);
    return () => clearInterval(timer);
  }, []);

  const isOnline = engineStatus === 'online';

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const res = await fetch('/analytics/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format })
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `atos_traffic_analytics.${format}`;
        a.click();
      }
    } catch (e) {
      alert('Failed to export report');
    }
  };

  const classData = [
    { name: 'Cars', count: isOnline ? Math.floor(telemetry.vehicles * 0.65) : 0, color: '#00f2fe' },
    { name: 'Motorcycles', count: isOnline ? Math.floor(telemetry.vehicles * 0.15) : 0, color: '#ff9f43' },
    { name: 'Buses', count: isOnline ? Math.floor(telemetry.vehicles * 0.10) : 0, color: '#4facfe' },
    { name: 'Trucks', count: isOnline ? Math.floor(telemetry.vehicles * 0.10) : 0, color: '#ff5252' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      {/* Action Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity color="var(--accent-cyan)" />
            REAL-TIME TRAFFIC ANALYTICS & EXPORT ENGINE
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Live vehicle volume time-series, pressure timeline, and automated report exporters
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-secondary" onClick={() => handleExport('csv')}>
            <Download size={14} /> Export CSV Report
          </button>
          <button className="btn-secondary" onClick={() => handleExport('json')}>
            <FileText size={14} /> Export JSON Data
          </button>
        </div>
      </div>

      {/* Top Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>ACTIVE VEHICLES</span>
            <Car size={18} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
            {isOnline ? telemetry.vehicles : '0'}
          </div>
          <div style={{ fontSize: '0.75rem', color: isOnline ? 'var(--accent-green)' : 'var(--text-muted)', marginTop: '4px' }}>
            {isOnline ? 'Real-time edge detection active' : 'Waiting for Engine Stream'}
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>TRAFFIC PRESSURE INDEX</span>
            <Activity size={18} color="var(--accent-green)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
            {isOnline ? telemetry.pressure.toFixed(2) : '0.00'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Capacity Threshold: 200 vehicles
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>PIPELINE LATENCY</span>
            <Clock size={18} color="var(--accent-orange)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>
            {isOnline ? `${telemetry.latency_ms.toFixed(1)} ms` : '--'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            CUDA Preprocess + TensorRT FP16
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>INCIDENT ALERTS</span>
            <ShieldAlert size={18} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-purple)' }}>
            {telemetry.alerts.length}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Active Incident Alerts Queue
          </div>
        </div>
      </div>

      {/* Main Charts Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-main)' }}>
              REAL-TIME TRAFFIC PRESSURE & FLOW TIMELINE
            </h3>
            <span className="badge badge-cyan">1 SEC SAMPLING</span>
          </div>

          <div style={{ height: '300px', width: '100%' }}>
            {historyData.length === 0 ? (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Waiting for Engine Telemetry Buffer...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={historyData}>
                  <defs>
                    <linearGradient id="pressureGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f2fe" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#00f2fe" stopOpacity={0.0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2638" />
                  <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} domain={[0, 1.0]} />
                  <Tooltip contentStyle={{ background: '#0f131d', border: '1px solid #1e2638', borderRadius: '8px', fontSize: '0.8rem' }} />
                  <Area type="monotone" dataKey="pressure" stroke="#00f2fe" strokeWidth={2} fillOpacity={1} fill="url(#pressureGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-main)' }}>
            VEHICLE CLASS BREAKDOWN
          </h3>

          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2638" />
                <XAxis type="number" stroke="#64748b" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={80} />
                <Tooltip contentStyle={{ background: '#0f131d', border: '1px solid #1e2638', borderRadius: '8px', fontSize: '0.8rem' }} />
                <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
