export type ActiveTab = 
  | 'grid' 
  | 'analytics' 
  | 'signal' 
  | 'automation' 
  | 'twin' 
  | 'plugins' 
  | 'assistant' 
  | 'settings';

export interface CameraNode {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'degraded';
  fps: number;
  latencyMs: number;
  vehiclesCount: number;
  type: 'RTSP' | 'USB' | 'WEBCAM' | 'ONVIF' | 'DRONE';
  url: string;
  isBrowserCam?: boolean;
}

export interface TelemetryData {
  pressure: number;
  signal_phase: number;
  vehicles: number;
  fps: number;
  latency_ms: number;
  active_cameras: number;
  alerts: string[];
}

export interface PluginItem {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: string;
  version: string;
  installed: boolean;
  downloads: string;
  author: string;
}

export interface WorkflowNode {
  id: string;
  type: 'trigger' | 'model' | 'filter' | 'condition' | 'action';
  title: string;
  subtitle: string;
  icon: string;
}
