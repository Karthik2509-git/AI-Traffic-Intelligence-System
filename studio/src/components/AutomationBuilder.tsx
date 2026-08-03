import React from 'react';
import { 
  ReactFlow, 
  Controls, 
  Background, 
  applyNodeChanges, 
  applyEdgeChanges, 
  addEdge,
  type Node, 
  type Edge, 
  type OnNodesChange, 
  type OnEdgesChange, 
  type OnConnect 
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Zap, Play } from 'lucide-react';

const initialNodes: Node[] = [
  {
    id: 'node-1',
    type: 'input',
    data: { label: '📹 Camera Stream (RTSP / USB)' },
    position: { x: 50, y: 150 },
    style: { background: '#0e1320', color: '#fff', border: '1px solid #00f2fe', borderRadius: '8px', padding: '12px' }
  },
  {
    id: 'node-2',
    data: { label: '⚡ TensorRT YOLOv8 (FP16 Engine)' },
    position: { x: 300, y: 150 },
    style: { background: '#0e1320', color: '#fff', border: '1px solid #a855f7', borderRadius: '8px', padding: '12px' }
  },
  {
    id: 'node-3',
    data: { label: '🔍 Filter: Class = Truck OR Bus' },
    position: { x: 580, y: 80 },
    style: { background: '#0e1320', color: '#fff', border: '1px solid #ff9f43', borderRadius: '8px', padding: '12px' }
  },
  {
    id: 'node-4',
    data: { label: '🚦 Condition: Density > 80%' },
    position: { x: 580, y: 220 },
    style: { background: '#0e1320', color: '#fff', border: '1px solid #f59e0b', borderRadius: '8px', padding: '12px' }
  },
  {
    id: 'node-5',
    type: 'output',
    data: { label: '📢 Telegram Dispatcher Alert' },
    position: { x: 860, y: 150 },
    style: { background: '#0e1320', color: '#fff', border: '1px solid #00ff9d', borderRadius: '8px', padding: '12px' }
  },
];

const initialEdges: Edge[] = [
  { id: 'e1-2', source: 'node-1', target: 'node-2', animated: true, style: { stroke: '#00f2fe' } },
  { id: 'e2-3', source: 'node-2', target: 'node-3', animated: true, style: { stroke: '#a855f7' } },
  { id: 'e2-4', source: 'node-2', target: 'node-4', animated: true, style: { stroke: '#a855f7' } },
  { id: 'e3-5', source: 'node-3', target: 'node-5', animated: true, style: { stroke: '#00ff9d' } },
  { id: 'e4-5', source: 'node-4', target: 'node-5', animated: true, style: { stroke: '#00ff9d' } },
];

export const AutomationBuilder: React.FC = () => {
  const [nodes, setNodes] = React.useState<Node[]>(initialNodes);
  const [edges, setEdges] = React.useState<Edge[]>(initialEdges);
  const [executing, setExecuting] = React.useState<boolean>(false);

  const onNodesChange: OnNodesChange = React.useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  );
  const onEdgesChange: OnEdgesChange = React.useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );
  const onConnect: OnConnect = React.useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [],
  );

  const handleTestRun = () => {
    setExecuting(true);
    setTimeout(() => setExecuting(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Zap color="var(--accent-cyan)" />
            REACT FLOW AUTOMATION BUILDER (n8n Style)
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Interactive node-based visual workflow editor connecting AI detections to webhooks, Telegram, & signal controllers
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-primary" onClick={handleTestRun}>
            <Play size={14} /> {executing ? 'Executing Flow...' : 'Test Run Workflow'}
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{ flex: 1, minHeight: '480px', borderRadius: '12px', overflow: 'hidden' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        >
          <Background color="#1e2638" gap={20} />
          <Controls style={{ background: '#0a0d16', color: '#fff', border: '1px solid #1e2638' }} />
        </ReactFlow>
      </div>
    </div>
  );
};
