import React from 'react';
import { Zap, Plus, Play, ArrowRight } from 'lucide-react';
import type { WorkflowNode } from '../types';

export const AutomationBuilder: React.FC = () => {
  const [nodes] = React.useState<WorkflowNode[]>([
    { id: '1', type: 'trigger', title: 'Camera Stream Input', subtitle: 'Intersection-Alpha (RTSP)', icon: 'camera' },
    { id: '2', type: 'model', title: 'TensorRT YOLOv8 Engine', subtitle: 'FP16 Detection Model', icon: 'cpu' },
    { id: '3', type: 'filter', title: 'Class Filter', subtitle: 'Class = Truck OR Bus', icon: 'filter' },
    { id: '4', type: 'condition', title: 'Density Check', subtitle: 'Queue Pressure > 80%', icon: 'activity' },
    { id: '5', type: 'action', title: 'Telegram Dispatcher', subtitle: 'Broadcast Incident Alert', icon: 'send' },
  ]);

  const [executing, setExecuting] = React.useState<boolean>(false);

  const runWorkflow = () => {
    setExecuting(true);
    setTimeout(() => setExecuting(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Zap color="var(--accent-cyan)" />
            WORKFLOW AUTOMATION ENGINE (n8n Style)
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Visual drag-and-drop workflow automation connecting AI detections to instant webhooks & alerts
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-secondary">
            <Plus size={14} /> + Add Node
          </button>
          <button className="btn-primary" onClick={runWorkflow}>
            <Play size={14} /> {executing ? 'Executing Pipeline...' : 'Test Run Workflow'}
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{
        padding: '30px',
        flex: 1,
        minHeight: '400px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'radial-gradient(circle at center, #0e1320 0%, #07090e 100%)',
        position: 'relative',
        overflowX: 'auto'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '30px' }}>
          {nodes.map((node, index) => (
            <React.Fragment key={node.id}>
              <div
                className="glass-panel-interactive"
                style={{
                  width: '220px',
                  padding: '16px',
                  borderLeft: '4px solid ' + (
                    node.type === 'trigger' ? 'var(--accent-cyan)' :
                    node.type === 'model' ? 'var(--accent-purple)' :
                    node.type === 'filter' ? 'var(--accent-orange)' :
                    node.type === 'condition' ? '#f59e0b' : 'var(--accent-green)'
                  )
                }}
              >
                <div style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
                  {node.type}
                </div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', marginBottom: '4px' }}>
                  {node.title}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {node.subtitle}
                </div>

                <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.7rem' }}>
                  <span className="badge badge-cyan">ACTIVE</span>
                  <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>0.2ms</span>
                </div>
              </div>

              {index < nodes.length - 1 && (
                <div style={{ display: 'flex', alignItems: 'center', color: 'var(--accent-cyan)' }}>
                  <ArrowRight size={24} />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};
