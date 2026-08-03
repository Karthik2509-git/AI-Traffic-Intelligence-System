import React from 'react';
import { Smartphone, Video, X } from 'lucide-react';

interface BrowserCamModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConnectStream: (stream: MediaStream) => void;
}

export const BrowserCamModal: React.FC<BrowserCamModalProps> = ({
  isOpen,
  onClose,
  onConnectStream
}) => {
  const [error, setError] = React.useState<string | null>(null);

  if (!isOpen) return null;

  const handleStartCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: false
      });
      onConnectStream(stream);
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to access camera. Please check browser permissions.');
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(5, 7, 12, 0.85)',
      backdropFilter: 'blur(10px)',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        width: '460px',
        padding: '28px',
        borderRadius: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
        position: 'relative'
      }}>
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer'
          }}
        >
          <X size={20} />
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #00f2fe, #4facfe)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#000'
          }}>
            <Smartphone size={24} />
          </div>

          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff' }}>
              Connect Browser Camera
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Turn your smartphone or laptop webcam into a live AI sensor node
            </p>
          </div>
        </div>

        <div style={{
          background: '#070a10',
          border: '1px solid var(--border-dim)',
          padding: '16px',
          borderRadius: '10px',
          fontSize: '0.82rem',
          color: 'var(--text-dim)',
          lineHeight: '1.6'
        }}>
          <div>✨ <strong>Zero Application Required:</strong> Works directly inside modern web browsers.</div>
          <div>🔒 <strong>Edge Processing:</strong> Frames are ingested and analyzed locally with zero cloud streaming.</div>
        </div>

        {error && (
          <div style={{
            background: '#1a1012',
            border: '1px solid #3d1c24',
            padding: '10px 14px',
            borderRadius: '6px',
            fontSize: '0.8rem',
            color: 'var(--accent-red)'
          }}>
            ⚠️ {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-primary" onClick={handleStartCamera}>
            <Video size={16} /> Allow Camera & Start Stream
          </button>
        </div>
      </div>
    </div>
  );
};
