import React from 'react';
import type { CameraNode, TelemetryData } from '../types';
import { Video } from 'lucide-react';

interface CameraGridProps {
  cameras: CameraNode[];
  telemetry: TelemetryData;
  onOpenMobileCam: () => void;
  browserStream: MediaStream | null;
}

export const CameraGrid: React.FC<CameraGridProps> = ({
  cameras,
  telemetry,
  onOpenMobileCam,
  browserStream
}) => {
  const [gridSize, setGridSize] = React.useState<number>(4);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);

  React.useEffect(() => {
    if (videoRef.current && browserStream) {
      videoRef.current.srcObject = browserStream;
    }
  }, [browserStream]);

  const displayedCameras = cameras.slice(0, gridSize);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
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
                color: gridSize === size ? '#000' : 'var(--text-main)',
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

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button className="btn-secondary" onClick={onOpenMobileCam}>
            <Video size={14} />
            + Add Browser Cam
          </button>
        </div>
      </div>

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
                backdropFilter: 'blur(8px)',
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
                  background: cam.status === 'online' ? 'var(--accent-green)' : 'var(--accent-orange)'
                }}></span>
                <span style={{ fontWeight: 700, color: '#fff' }}>{cam.name}</span>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>({cam.type})</span>
              </div>

              <div style={{
                background: 'rgba(10, 13, 22, 0.85)',
                backdropFilter: 'blur(8px)',
                padding: '4px 10px',
                borderRadius: '6px',
                border: '1px solid var(--border-dim)',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent-cyan)'
              }}>
                {cam.fps.toFixed(1)} FPS • {cam.latencyMs.toFixed(1)}ms
              </div>
            </div>

            {cam.isBrowserCam && browserStream ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
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
                    car 0.94
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
                    bus 0.88
                  </span>
                </div>

                <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                  <line x1="10%" y1="90%" x2="45%" y2="40%" stroke="rgba(0, 242, 254, 0.4)" strokeWidth="2" strokeDasharray="6,6" />
                  <line x1="90%" y1="90%" x2="55%" y2="40%" stroke="rgba(0, 242, 254, 0.4)" strokeWidth="2" strokeDasharray="6,6" />
                  <polygon points="45,40 55,40 90,90 10,90" fill="rgba(0, 242, 254, 0.04)" />
                </svg>
              </div>
            )}

            <div style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              right: '12px',
              background: 'rgba(10, 13, 22, 0.85)',
              backdropFilter: 'blur(8px)',
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
                Active Detections: <strong style={{ color: '#fff' }}>{idx === 0 ? telemetry.vehicles : Math.floor(cam.vehiclesCount)}</strong>
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
