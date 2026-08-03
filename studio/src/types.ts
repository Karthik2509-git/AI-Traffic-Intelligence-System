export type ActiveTab = 
  | 'grid' 
  | 'analytics' 
  | 'signal' 
  | 'automation' 
  | 'twin' 
  | 'plugins' 
  | 'assistant' 
  | 'settings'
  | 'health'
  | 'replay'
  | 'logs';

export type UserRole = 'Guest' | 'Developer' | 'Administrator';

export interface CameraNode {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'waiting_for_engine';
  fps: number;
  latency_ms: number;
  vehiclesCount?: number;
  type: 'RTSP' | 'USB' | 'WEBCAM' | 'ONVIF' | 'LOCAL_FILE';
  url: string;
  resolution?: string;
  dropped_frames?: number;
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

export interface EngineMetrics {
  cuda_device: string;
  cuda_status: string;
  tensorrt_version: string;
  precision: string;
  gpu_utilization_pct: number;
  vram_used_mb: number;
  vram_total_mb: number;
  cpu_utilization_pct?: number;
  ram_utilization_pct?: number;
  queue_depth: number;
  dropped_frames: number;
  inference_ms: number;
  preprocess_ms: number;
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

export interface SystemLog {
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
  message: string;
}
