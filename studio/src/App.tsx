import React from 'react';
import type { ActiveTab, CameraNode, TelemetryData, EngineMetrics, UserRole } from './types';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { CameraGrid } from './components/CameraGrid';
import { TrafficAnalytics } from './components/TrafficAnalytics';
import { SignalControllerView } from './components/SignalControllerView';
import { AutomationBuilder } from './components/AutomationBuilder';
import { DigitalTwin3D } from './components/DigitalTwin3D';
import { PluginStore } from './components/PluginStore';
import { AIAssistant } from './components/AIAssistant';
import { EngineSettings } from './components/EngineSettings';
import { HealthDashboard } from './components/HealthDashboard';
import { ReplaySystem } from './components/ReplaySystem';
import { LogViewer } from './components/LogViewer';
import { BrowserCamModal } from './components/BrowserCamModal';

export function App() {
  const [activeTab, setActiveTab] = React.useState<ActiveTab>('grid');
  const [userRole, setUserRole] = React.useState<UserRole>('Administrator');
  const [isBrowserCamOpen, setIsBrowserCamOpen] = React.useState<boolean>(false);
  const [browserStream, setBrowserStream] = React.useState<MediaStream | null>(null);

  const [engineStatus, setEngineStatus] = React.useState<string>('waiting_for_engine');

  const [telemetry, setTelemetry] = React.useState<TelemetryData>({
    pressure: 0.0,
    signal_phase: 0,
    vehicles: 0,
    fps: 0.0,
    latency_ms: 0.0,
    active_cameras: 1,
    alerts: []
  });

  const [metrics, setMetrics] = React.useState<EngineMetrics>({
    cuda_device: 'NVIDIA GeForce RTX 4090 / CUDA 12.4',
    cuda_status: 'Ready',
    tensorrt_version: '10.0.1',
    precision: 'FP16',
    gpu_utilization_pct: 0,
    vram_used_mb: 0,
    vram_total_mb: 16384,
    cpu_utilization_pct: 0,
    ram_utilization_pct: 0,
    queue_depth: 0,
    dropped_frames: 0,
    inference_ms: 0.0,
    preprocess_ms: 0.0
  });

  const [cameras, setCameras] = React.useState<CameraNode[]>([
    { id: 'cam-1', name: 'Intersection Alpha North', location: 'Main Arterial St.', status: 'waiting_for_engine', fps: 0.0, latency_ms: 0.0, vehiclesCount: 0, type: 'RTSP', url: 'rtsp://192.168.1.100/stream' },
    { id: 'cam-2', name: 'Intersection Alpha East', location: '5th Ave & Main', status: 'waiting_for_engine', fps: 0.0, latency_ms: 0.0, vehiclesCount: 0, type: 'USB', url: '0' },
    { id: 'cam-3', name: 'South Transit Corridor', location: 'South Gate Highway', status: 'waiting_for_engine', fps: 0.0, latency_ms: 0.0, vehiclesCount: 0, type: 'RTSP', url: 'rtsp://192.168.1.102/stream' },
    { id: 'cam-4', name: 'West Commercial Zone', location: 'Retail Center Blvd.', status: 'waiting_for_engine', fps: 0.0, latency_ms: 0.0, vehiclesCount: 0, type: 'ONVIF', url: 'rtsp://192.168.1.104/stream' },
  ]);

  // Real WebSocket & REST Telemetry Connection Loop
  React.useEffect(() => {
    let ws: WebSocket | null = null;

    const connectWebSocket = () => {
      try {
        const wsUrl = `ws://${window.location.hostname}:8080/ws/telemetry`;
        ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
          try {
            const snapshot = JSON.parse(event.data);
            if (snapshot.engine_status) setEngineStatus(snapshot.engine_status);
            if (snapshot.telemetry) setTelemetry(snapshot.telemetry);
            if (snapshot.metrics) setMetrics(snapshot.metrics);
            if (snapshot.cameras && snapshot.cameras.length > 0) setCameras(snapshot.cameras);
          } catch (err) {
            console.error('WS parse error:', err);
          }
        };

        ws.onclose = () => {
          setTimeout(connectWebSocket, 3000);
        };
      } catch (err) {
        console.log('WS Connection error, falling back to HTTP poll');
      }
    };

    connectWebSocket();

    const fetchHTTPFallback = async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          setEngineStatus(data.engine_status || 'waiting_for_engine');
        }
      } catch (e) {
        setEngineStatus('waiting_for_engine');
      }
    };

    const interval = setInterval(fetchHTTPFallback, 3000);

    return () => {
      if (ws) ws.close();
      clearInterval(interval);
    };
  }, []);

  const handleConnectBrowserStream = (stream: MediaStream) => {
    setBrowserStream(stream);
    const newCam: CameraNode = {
      id: 'cam-browser',
      name: 'Mobile Browser Node (Live)',
      location: 'Browser Camera Node',
      status: 'online',
      fps: 30.0,
      latency_ms: 8.5,
      vehiclesCount: 4,
      type: 'WEBCAM',
      url: 'browser://stream',
      isBrowserCam: true
    };
    setCameras((prev) => [newCam, ...prev]);
  };

  const refreshEngineMetrics = async () => {
    try {
      const res = await fetch('/engine');
      if (res.ok) {
        const data = await res.json();
        setMetrics(data.metrics || metrics);
      }
    } catch (e) {
      console.log('Engine metrics fetch');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-dark)' }}>
      {/* Top Navbar */}
      <Navbar
        telemetry={telemetry}
        engineStatus={engineStatus}
        userRole={userRole}
        setUserRole={setUserRole}
        onOpenMobileCam={() => setIsBrowserCamOpen(true)}
      />

      {/* Main App Layout */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Sidebar Navigation */}
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          engineStatus={engineStatus}
        />

        {/* Viewport Content */}
        <main style={{ flex: 1, padding: '20px', overflowY: 'auto', background: 'var(--bg-dark)' }}>
          {activeTab === 'grid' && (
            <CameraGrid
              cameras={cameras}
              telemetry={telemetry}
              engineStatus={engineStatus}
              onOpenMobileCam={() => setIsBrowserCamOpen(true)}
              browserStream={browserStream}
            />
          )}

          {activeTab === 'analytics' && (
            <TrafficAnalytics telemetry={telemetry} engineStatus={engineStatus} />
          )}

          {activeTab === 'signal' && (
            <SignalControllerView telemetry={telemetry} />
          )}

          {activeTab === 'automation' && (
            <AutomationBuilder />
          )}

          {activeTab === 'twin' && (
            <DigitalTwin3D telemetry={telemetry} />
          )}

          {activeTab === 'plugins' && (
            <PluginStore />
          )}

          {activeTab === 'assistant' && (
            <AIAssistant telemetry={telemetry} engineStatus={engineStatus} />
          )}

          {activeTab === 'health' && (
            <HealthDashboard metrics={metrics} engineStatus={engineStatus} onRefresh={refreshEngineMetrics} />
          )}

          {activeTab === 'replay' && (
            <ReplaySystem />
          )}

          {activeTab === 'logs' && (
            <LogViewer />
          )}

          {activeTab === 'settings' && (
            <EngineSettings />
          )}
        </main>
      </div>

      {/* Mobile/Browser Camera Pairing Modal */}
      <BrowserCamModal
        isOpen={isBrowserCamOpen}
        onClose={() => setIsBrowserCamOpen(false)}
        onConnectStream={handleConnectBrowserStream}
      />
    </div>
  );
}

export default App;
