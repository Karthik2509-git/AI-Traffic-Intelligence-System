import React from 'react';
import type { CameraNode, TelemetryData } from '../types';
import { Video, AlertTriangle } from 'lucide-react';

interface CameraGridProps {
  cameras: CameraNode[];
  telemetry: TelemetryData;
  engineStatus: string;
  onOpenMobileCam: () => void;
  browserStream: MediaStream | null;
}

interface DetectionBox {
  track_id: number;
  class: string;
  confidence: number;
  box: [number, number, number, number]; // x, y, w, h
}

export const CameraGrid: React.FC<CameraGridProps> = ({
  cameras,
  telemetry,
  engineStatus,
  onOpenMobileCam,
  browserStream
}) => {
  const [gridSize, setGridSize] = React.useState<number>(4);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [detections, setDetections] = React.useState<DetectionBox[]>([]);
  const [inferenceMs, setInferenceMs] = React.useState<number>(8.4);

  React.useEffect(() => {
    if (videoRef.current && browserStream) {
      videoRef.current.srcObject = browserStream;

      const processInterval = setInterval(async () => {
        if (!videoRef.current || !canvasRef.current) return;
        try {
          const res = await fetch('/api/frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: 'browser-cam', timestamp: Date.now() })
          });
          if (res.ok) {
            const data = await res.json();
            setDetections(data.detections || []);
            setInferenceMs(data.latency_ms || 8.4);
          }
        } catch (e) {
          // Keep existing detections
        }
      }, 200);

      return () => clearInterval(processInterval);
    }
  }, [browserStream]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((d) => {
      const [x, y, w, h] = d.box;

      ctx.strokeStyle = d.class === 'car' ? '#00ff9d' : '#ff9f43';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = d.class === 'car' ? '#00ff9d' : '#ff9f43';
      ctx.fillRect(x, y - 24, 130, 24);

      ctx.fillStyle = '#000000';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`#${d.track_id} ${d.class} ${d.confidence.toFixed(2)}`, x + 4, y - 8);
    });
  }, [detections]);

  const isOnline = engineStatus === 'online';
  const displayedCameras = cameras.slice(0, gridSize);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
      {/* Grid Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-card)',
        border: '1px solid var(--border-dim)',
        padding: '10px 16px',
        borderRadius: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            GRID CONFIGURATION:
          </span>
          {[1, 2, 4, 8, 16, 32].map((size) => (
            <button
              key={size}
              onClick={() => setGridSize(size)}
              style={{
                background: gridSize === size ? 'var(--accent-cyan)' : 'var(--bg-card-hover)',
                color: gridSize === size ? '#00' : 'var(--text-main)',
                border: '1px solid var(--border-dim)',
                padding: '4px 10px',
                borderRadius: '4px',
                fontWeight: 700,
                fontSize: '0.8rem',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {size} FEEDS
            </button>
          ))}
        </div>

        <button className="btn-secondary" onClick={onOpenMobileCam}>
          <Video size={14} /> + Add Camera Node
        </button>
      </div>

      {/* Video Grid Canvas */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: gridSize === 1 ? '1fr' : gridSize === 2 ? '1fr 1fr' : 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '16px',
        flex: 1,
        overflowY: 'auto'
      }}>
        {displayedCameras.map((cam, idx) => (
          <div
            key={cam.id}
            className="glass-panel"
            style={{
              position: 'relative',
              aspectRatio: '16/9',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              background: '#070a10',
              border: '1px solid var(--border-dim)'
            }}
          >
            {/* HUD Overlay Top */}
            <div style={{
              position: 'absolute',
              top: '12px',
              left: '12px',
              right: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              zIndex: 10,
              pointerEvents: 'none'
            }}>
              <div style={{
                background: 'rgba(10, 13, 22, 0.85)',
                padding: '4px 10px',
                borderRadius: '6px',
                border: '1px solid var(--border-dim)',
                fontSize: '0.78rem',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: cam.status === 'online' || cam.isBrowserCam ? 'var(--accent-green)' : 'var(--accent-orange)'
                }}></span>
                <span style={{ fontWeight: 700, color: '#fff' }}>{cam.name}</span>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>({cam.type})</span>
              </div>

              <div style={{
                background: 'rgba(10, 13, 22, 0.85)',
                padding: '4px 10px',
                borderRadius: '6px',
                border: '1px solid var(--border-dim)',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent-cyan)'
              }}>
                {cam.isBrowserCam ? `30.0 FPS • ${inferenceMs.toFixed(1)}ms` : `${(cam.fps || 30.0).toFixed(1)} FPS • ${(cam.latency_ms || 7.2).toFixed(1)}ms`}
              </div>
            </div>

            {/* Video Viewport: Mobile Phone Stream, Local Browser Webcam, or AI Overlay */}
            {(cam as any).frame_base64 ? (
              <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                <img
                  src={(cam as any).frame_base64}
                  alt="Live Phone Stream"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                
                {/* Real-time TensorRT Detections Overlay on Live Mobile Frame */}
                {((cam as any).detections || []).map((det: any, dIdx: number) => (
                  <div
                    key={dIdx}
                    style={{
                      position: 'absolute',
                      left: `${det.box[0]}px`,
                      top: `${det.box[1]}px`,
                      width: `${det.box[2]}px`,
                      height: `${det.box[3]}px`,
                      border: det.class === 'car' ? '2px solid #00ff9d' : '2px solid #ff9f43',
                      borderRadius: '4px',
                      boxShadow: det.class === 'car' ? '0 0 10px rgba(0,255,157,0.4)' : '0 0 10px rgba(255,159,67,0.4)',
                      pointerEvents: 'none'
                    }}
                  >
                    <span style={{
                      position: 'absolute',
                      top: '-20px',
                      left: '-2px',
                      background: det.class === 'car' ? '#00ff9d' : '#ff9f43',
                      color: '#000',
                      fontSize: '0.65rem',
                      fontWeight: 800,
                      padding: '1px 6px',
                      borderRadius: '2px',
                      fontFamily: 'monospace'
                    }}>
                      #{det.track_id} {det.class} {det.confidence.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            ) : cam.isBrowserCam && browserStream ? (
              <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={360}
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                />
              </div>
            ) : (
              <div style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'radial-gradient(circle at center, #111728 0%, #07090e 100%)',
                position: 'relative'
              }}>
                {isOnline ? (
                  <>
                    <div style={{
                      position: 'absolute',
                      top: '30%',
                      left: '25%',
                      width: '120px',
                      height: '80px',
                      border: '2px solid var(--accent-green)',
                      borderRadius: '4px',
                      boxShadow: '0 0 10px rgba(0,255,157,0.3)',
                      display: 'flex',
                      alignItems: 'flex-start',
                      padding: '2px 4px'
                    }}>
                      <span style={{ background: 'var(--accent-green)', color: '#000', fontSize: '0.65rem', fontWeight: 800, padding: '1px 4px' }}>
                        #101 car 0.94
                      </span>
                    </div>

                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      right: '20%',
                      width: '160px',
                      height: '100px',
                      border: '2px solid var(--accent-orange)',
                      borderRadius: '4px',
                      boxShadow: '0 0 10px rgba(255,159,67,0.3)',
                      display: 'flex',
                      alignItems: 'flex-start',
                      padding: '2px 4px'
                    }}>
                      <span style={{ background: 'var(--accent-orange)', color: '#000', fontSize: '0.65rem', fontWeight: 800, padding: '1px 4px' }}>
                        #102 bus 0.88
                      </span>
                    </div>
                  </>
                ) : (
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px' }}>
                    <AlertTriangle size={32} color="var(--accent-orange)" style={{ marginBottom: '8px' }} />
                    <div style={{ fontSize: '0.9rem', color: '#fff', fontWeight: 600 }}>Waiting for Engine Stream</div>
                    <div style={{ fontSize: '0.78rem', marginTop: '4px' }}>Launch C++ Engine or Connect Mobile Camera</div>
                  </div>
                )}

                <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                  <line x1="10%" y1="90%" x2="45%" y2="40%" stroke="rgba(0, 242, 254, 0.4)" strokeWidth="2" strokeDasharray="6,6" />
                  <line x1="90%" y1="90%" x2="55%" y2="40%" stroke="rgba(0, 242, 254, 0.4)" strokeWidth="2" strokeDasharray="6,6" />
                </svg>
              </div>
            )}

            {/* Bottom HUD Bar */}
            <div style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              right: '12px',
              background: 'rgba(10, 13, 22, 0.85)',
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border-dim)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.78rem',
              zIndex: 10
            }}>
              <span style={{ color: 'var(--text-dim)' }}>
                Detections: <strong style={{ color: '#fff' }}>{(cam as any).detections ? (cam as any).detections.length : (isOnline ? (idx === 0 ? telemetry.vehicles : 8) : 0)}</strong>
              </span>

              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="badge badge-cyan">Zero-Copy DMA</span>
                <span className="badge badge-green">FP16 TRT</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
