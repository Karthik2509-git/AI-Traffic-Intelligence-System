import React from 'react';
import type { ActiveTab, CameraNode, TelemetryData } from './types';
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
import { BrowserCamModal } from './components/BrowserCamModal';

export function App() {
  const [activeTab, setActiveTab] = React.useState<ActiveTab>('grid');
  const [isBrowserCamOpen, setIsBrowserCamOpen] = React.useState<boolean>(false);
  const [browserStream, setBrowserStream] = React.useState<MediaStream | null>(null);

  const [telemetry, setTelemetry] = React.useState<TelemetryData>({
    pressure: 0.18,
    signal_phase: 0,
    vehicles: 18,
    fps: 45.2,
    latency_ms: 12.4,
    active_cameras: 4,
    alerts: []
  });

  const [cameras, setCameras] = React.useState<CameraNode[]>([
    { id: 'cam-1', name: 'Intersection Alpha North', location: 'Main Arterial St.', status: 'online', fps: 45.2, latencyMs: 12.4, vehiclesCount: 18, type: 'RTSP', url: 'rtsp://192.168.1.100/stream' },
    { id: 'cam-2', name: 'Intersection Alpha East', location: '5th Ave & Main', status: 'online', fps: 44.8, latencyMs: 13.1, vehiclesCount: 12, type: 'USB', url: '0' },
    { id: 'cam-3', name: 'South Transit Corridor', location: 'South Gate Highway', status: 'online', fps: 46.0, latencyMs: 11.8, vehiclesCount: 8, type: 'RTSP', url: 'rtsp://192.168.1.102/stream' },
    { id: 'cam-4', name: 'West Commercial Zone', location: 'Retail Center Blvd.', status: 'online', fps: 43.5, latencyMs: 14.2, vehiclesCount: 14, type: 'ONVIF', url: 'rtsp://192.168.1.104/stream' },
  ]);

  React.useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch('/api/telemetry');
        if (res.ok) {
          const data = await res.json();
          setTelemetry((prev) => ({
            ...prev,
            pressure: data.pressure ?? prev.pressure,
            signal_phase: data.signal_phase ?? prev.signal_phase,
            vehicles: data.vehicles ?? prev.vehicles,
            fps: data.fps ?? prev.fps,
            latency_ms: data.latency_ms ?? prev.latency_ms,
            alerts: data.alerts ?? prev.alerts
          }));
        }
      } catch (e) {
        setTelemetry((prev) => ({
          ...prev,
          vehicles: Math.max(8, prev.vehicles + (Math.random() > 0.5 ? 1 : -1)),
          pressure: Math.min(1.0, Math.max(0.05, prev.pressure + (Math.random() - 0.5) * 0.02))
        }));
      }
    };

    const interval = setInterval(fetchTelemetry, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleConnectBrowserStream = (stream: MediaStream) => {
    setBrowserStream(stream);
    const newCam: CameraNode = {
      id: 'cam-browser',
      name: 'Mobile Browser Node (Live)',
      location: 'Browser Camera Node',
      status: 'online',
      fps: 30.0,
      latencyMs: 8.5,
      vehiclesCount: 4,
      type: 'WEBCAM',
      url: 'browser://stream',
      isBrowserCam: true
    };
    setCameras((prev) => [newCam, ...prev]);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-dark)' }}>
      <Navbar
        telemetry={telemetry}
        onOpenMobileCam={() => setIsBrowserCamOpen(true)}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main style={{ flex: 1, padding: '20px', overflowY: 'auto', background: 'var(--bg-dark)' }}>
          {activeTab === 'grid' && (
            <CameraGrid
              cameras={cameras}
              telemetry={telemetry}
              onOpenMobileCam={() => setIsBrowserCamOpen(true)}
              browserStream={browserStream}
            />
          )}

          {activeTab === 'analytics' && (
            <TrafficAnalytics telemetry={telemetry} />
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
            <AIAssistant telemetry={telemetry} />
          )}

          {activeTab === 'settings' && (
            <EngineSettings />
          )}
        </main>
      </div>

      <BrowserCamModal
        isOpen={isBrowserCamOpen}
        onClose={() => setIsBrowserCamOpen(false)}
        onConnectStream={handleConnectBrowserStream}
      />
    </div>
  );
}

export default App;
