import React from 'react';
import { Smartphone, Video, Wifi, Battery, Power, ShieldAlert } from 'lucide-react';

export const MobileCamNode: React.FC = () => {
  const [sessionId, setSessionId] = React.useState<string>('');
  const [isStreaming, setIsStreaming] = React.useState<boolean>(false);
  const [facingMode, setFacingMode] = React.useState<'environment' | 'user'>('environment');
  const [resolution, setResolution] = React.useState<string>('720p');
  const [fpsTarget, setFpsTarget] = React.useState<number>(30);
  const [batteryPct, setBatteryPct] = React.useState<number>(92);
  const [frameCount, setFrameCount] = React.useState<number>(0);
  const [latencyMs] = React.useState<number>(7.2);
  const [status, setStatus] = React.useState<string>('Ready to Stream');
  const [isSecureContext, setIsSecureContext] = React.useState<boolean>(true);

  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const wsRef = React.useRef<WebSocket | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);

  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const s = params.get('session') || `mob-${Math.random().toString(36).substring(2, 9)}`;
    setSessionId(s);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setIsSecureContext(false);
      setStatus('Secure Context (HTTPS) Required');
    }

    if ('getBattery' in navigator) {
      (navigator as any).getBattery().then((battery: any) => {
        setBatteryPct(Math.floor(battery.level * 100));
        battery.addEventListener('levelchange', () => {
          setBatteryPct(Math.floor(battery.level * 100));
        });
      });
    }
  }, []);

  const startCamera = async () => {
    try {
      setStatus('Accessing Camera...');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access requires HTTPS security permission.');
      }

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: facingMode,
          width: resolution === '1080p' ? 1920 : resolution === '720p' ? 1280 : 640,
          height: resolution === '1080p' ? 1080 : resolution === '720p' ? 720 : 480,
          frameRate: fpsTarget
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Connect WebSocket via same host origin (proxied by Vite)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream/${sessionId}`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('Streaming Live to Studio');
        setIsStreaming(true);
      };

      ws.onclose = () => {
        setStatus('Disconnected');
        setIsStreaming(false);
      };

      ws.onerror = (err) => {
        console.error('WS error:', err);
        setStatus('WebSocket Error (Reconnecting)');
      };

      const canvas = canvasRef.current || document.createElement('canvas');
      canvasRef.current = canvas;
      const ctx = canvas.getContext('2d');

      const intervalMs = 1000 / Math.min(fpsTarget, 15);
      const streamTimer = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN || !videoRef.current) return;

        canvas.width = 480;
        canvas.height = 270;
        ctx?.drawImage(videoRef.current, 0, 0, 480, 270);
        const base64Image = canvas.toDataURL('image/jpeg', 0.5);

        const payload = {
          session_id: sessionId,
          image: base64Image,
          timestamp: Date.now(),
          battery: batteryPct,
          resolution: resolution,
          fps: fpsTarget
        };

        ws.send(JSON.stringify(payload));
        setFrameCount((prev) => prev + 1);
      }, intervalMs);

      return () => clearInterval(streamTimer);
    } catch (err: any) {
      setStatus(`Error: ${err?.message || 'Camera permission denied'}`);
      setIsStreaming(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsStreaming(false);
    setStatus('Streaming Stopped');
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#07090e',
      color: '#fff',
      fontFamily: 'Inter, system-ui, sans-serif',
      padding: '16px'
    }}>
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: '#0b0f19',
        border: '1px solid #1e2638',
        padding: '12px 16px',
        borderRadius: '12px',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Smartphone color="var(--accent-cyan)" size={22} />
          <div>
            <div style={{ fontWeight: 800, fontSize: '0.95rem', letterSpacing: '0.5px' }}>
              ATOS MOBILE CAMERA NODE
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'monospace' }}>
              SESSION: {sessionId || 'INITIALIZING'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '0.78rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: batteryPct < 20 ? '#ff5252' : '#00ff9d' }}>
            <Battery size={16} /> {batteryPct}%
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#00f2fe' }}>
            <Wifi size={16} /> LAN
          </div>
        </div>
      </header>

      {!isSecureContext && (
        <div style={{
          background: 'rgba(255, 82, 82, 0.15)',
          border: '1px solid #ff5252',
          padding: '12px 16px',
          borderRadius: '10px',
          marginBottom: '16px',
          fontSize: '0.8rem',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          color: '#ff8a8a'
        }}>
          <ShieldAlert size={20} color="#ff5252" />
          <span>
            Mobile browsers require <strong>HTTPS</strong> for camera hardware access. Make sure your phone address starts with <strong>https://</strong>.
          </span>
        </div>
      )}

      <div style={{
        flex: 1,
        background: '#0a0d16',
        border: '1px solid #1e2638',
        borderRadius: '16px',
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />

        <div style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          right: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          pointerEvents: 'none'
        }}>
          <div style={{
            background: 'rgba(10, 13, 22, 0.85)',
            backdropFilter: 'blur(8px)',
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid #1e2638',
            fontSize: '0.78rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isStreaming ? '#00ff9d' : '#ff9f43',
              boxShadow: isStreaming ? '0 0 10px #00ff9d' : 'none'
            }} />
            <span style={{ fontWeight: 700 }}>{status}</span>
          </div>

          <div style={{
            background: 'rgba(10, 13, 22, 0.85)',
            backdropFilter: 'blur(8px)',
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid #1e2638',
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            color: '#00f2fe'
          }}>
            FRAMES: {frameCount} • {latencyMs.toFixed(1)}ms
          </div>
        </div>
      </div>

      <div style={{
        marginTop: '16px',
        background: '#0b0f19',
        border: '1px solid #1e2638',
        borderRadius: '16px',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          <div>
            <label style={{ fontSize: '0.72rem', color: '#64748b', display: 'block', marginBottom: '4px' }}>CAMERA</label>
            <select
              value={facingMode}
              onChange={(e) => setFacingMode(e.target.value as any)}
              disabled={isStreaming}
              style={{
                width: '100%',
                background: '#111726',
                border: '1px solid #1e2638',
                borderRadius: '6px',
                padding: '8px',
                color: '#fff',
                fontSize: '0.8rem',
                outline: 'none'
              }}
            >
              <option value="environment">Rear Camera</option>
              <option value="user">Front Camera</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.72rem', color: '#64748b', display: 'block', marginBottom: '4px' }}>RESOLUTION</label>
            <select
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              disabled={isStreaming}
              style={{
                width: '100%',
                background: '#111726',
                border: '1px solid #1e2638',
                borderRadius: '6px',
                padding: '8px',
                color: '#fff',
                fontSize: '0.8rem',
                outline: 'none'
              }}
            >
              <option value="480p">480p</option>
              <option value="720p">720p</option>
              <option value="1080p">1080p</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.72rem', color: '#64748b', display: 'block', marginBottom: '4px' }}>TARGET FPS</label>
            <select
              value={fpsTarget}
              onChange={(e) => setFpsTarget(parseInt(e.target.value))}
              disabled={isStreaming}
              style={{
                width: '100%',
                background: '#111726',
                border: '1px solid #1e2638',
                borderRadius: '6px',
                padding: '8px',
                color: '#fff',
                fontSize: '0.8rem',
                outline: 'none'
              }}
            >
              <option value={15}>15 FPS</option>
              <option value={30}>30 FPS</option>
              <option value={60}>60 FPS</option>
            </select>
          </div>
        </div>

        {!isStreaming ? (
          <button
            onClick={startCamera}
            style={{
              background: 'linear-gradient(135deg, #00f2fe, #4facfe)',
              color: '#000',
              fontWeight: 800,
              fontSize: '0.95rem',
              padding: '12px',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            <Video size={18} /> START STREAMING TO ATOS STUDIO
          </button>
        ) : (
          <button
            onClick={stopCamera}
            style={{
              background: '#ff5252',
              color: '#fff',
              fontWeight: 800,
              fontSize: '0.95rem',
              padding: '12px',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            <Power size={18} /> STOP STREAMING
          </button>
        )}
      </div>
    </div>
  );
};
