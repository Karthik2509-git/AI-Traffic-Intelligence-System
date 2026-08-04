import React from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Smartphone, Video, Usb, Wifi, Shield, FileVideo, X, QrCode } from 'lucide-react';

interface AddCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConnectBrowserCam: () => void;
  onConnectCustomCam: (name: string, type: string, url: string) => void;
}

export const AddCameraModal: React.FC<AddCameraModalProps> = ({
  isOpen,
  onClose,
  onConnectBrowserCam,
  onConnectCustomCam
}) => {
  const [selectedType, setSelectedType] = React.useState<string>('PHONE');
  const [selectedIp, setSelectedIp] = React.useState<string>('');
  const [allIps, setAllIps] = React.useState<{ interface: string; ip: string; is_virtual: boolean }[]>([]);
  const [sessionId, setSessionId] = React.useState<string>('');
  const [camName, setCamName] = React.useState<string>('Camera Node');
  const [camUrl, setCamUrl] = React.useState<string>('rtsp://192.168.1.100/stream');

  React.useEffect(() => {
    if (isOpen) {
      const uuidStr = `session-${Math.random().toString(36).substring(2, 9)}`;
      setSessionId(uuidStr);

      fetch('/api/local-ip')
        .then((res) => res.json())
        .then((data) => {
          if (data.local_ip) setSelectedIp(data.local_ip);
          if (data.all_ips) setAllIps(data.all_ips);
        })
        .catch(() => setSelectedIp('localhost'));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const phoneUrl = `http://${selectedIp || 'localhost'}:5173/mobile?session=${sessionId}`;

  const handleCustomConnect = () => {
    onConnectCustomCam(camName, selectedType, camUrl);
    onClose();
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
        width: '540px',
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
            <Video size={24} />
          </div>

          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff' }}>
              Add Camera Sensor Node
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Pair smartphones, IP webcams, RTSP streams, or USB cameras
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          {[
            { id: 'PHONE', label: 'Phone Camera', icon: <Smartphone size={16} /> },
            { id: 'BROWSER', label: 'Browser Cam', icon: <Video size={16} /> },
            { id: 'RTSP', label: 'RTSP Stream', icon: <Wifi size={16} /> },
            { id: 'USB', label: 'USB Camera', icon: <Usb size={16} /> },
            { id: 'ONVIF', label: 'ONVIF Camera', icon: <Shield size={16} /> },
            { id: 'FILE', label: 'Video File', icon: <FileVideo size={16} /> },
          ].map((type) => (
            <button
              key={type.id}
              onClick={() => setSelectedType(type.id)}
              style={{
                background: selectedType === type.id ? 'var(--accent-cyan)' : 'var(--bg-card-hover)',
                color: selectedType === type.id ? '#000' : 'var(--text-main)',
                border: '1px solid var(--border-dim)',
                borderRadius: '8px',
                padding: '10px',
                fontSize: '0.78rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                justifyContent: 'center'
              }}
            >
              {type.icon} {type.label}
            </button>
          ))}
        </div>

        {selectedType === 'PHONE' && (
          <div style={{
            background: '#070a10',
            border: '1px solid var(--border-dim)',
            padding: '20px',
            borderRadius: '12px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center',
            gap: '14px'
          }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <QrCode size={18} color="var(--accent-cyan)" />
              SCAN WITH YOUR MOBILE PHONE
            </div>

            {/* Wi-Fi Interface Selector if multiple exist */}
            {allIps.length > 1 && (
              <div style={{ width: '100%', maxWidth: '340px', textAlign: 'left' }}>
                <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  SELECT WI-FI NETWORK INTERFACE:
                </label>
                <select
                  value={selectedIp}
                  onChange={(e) => setSelectedIp(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#111726',
                    border: '1px solid var(--border-dim)',
                    borderRadius: '6px',
                    padding: '6px 10px',
                    color: '#fff',
                    fontSize: '0.8rem',
                    outline: 'none'
                  }}
                >
                  {allIps.map((item, idx) => (
                    <option key={idx} value={item.ip}>
                      {item.ip} ({item.interface})
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div style={{ background: '#fff', padding: '12px', borderRadius: '12px', display: 'inline-block' }}>
              <QRCodeSVG value={phoneUrl} size={160} />
            </div>

            <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
              {phoneUrl}
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Ensure your phone is connected to the same Wi-Fi network ({selectedIp})
            </div>
          </div>
        )}

        {selectedType === 'BROWSER' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
              Instantly connect your desktop or laptop webcam into ATOS Engine.
            </div>
            <button className="btn-primary" onClick={() => { onConnectBrowserCam(); onClose(); }}>
              <Video size={16} /> Connect Local Web Camera
            </button>
          </div>
        )}

        {(selectedType === 'RTSP' || selectedType === 'USB' || selectedType === 'ONVIF' || selectedType === 'FILE') && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                Camera Name
              </label>
              <input
                type="text"
                value={camName}
                onChange={(e) => setCamName(e.target.value)}
                style={{
                  width: '100%',
                  background: '#07090e',
                  border: '1px solid var(--border-dim)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  color: '#fff',
                  fontSize: '0.85rem'
                }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                Source URI / Path
              </label>
              <input
                type="text"
                value={camUrl}
                onChange={(e) => setCamUrl(e.target.value)}
                style={{
                  width: '100%',
                  background: '#07090e',
                  border: '1px solid var(--border-dim)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  color: '#fff',
                  fontSize: '0.85rem',
                  fontFamily: 'var(--font-mono)'
                }}
              />
            </div>

            <button className="btn-primary" onClick={handleCustomConnect} style={{ marginTop: '8px' }}>
              Add {selectedType} Camera Node
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
