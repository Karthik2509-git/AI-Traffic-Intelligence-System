import React from 'react';
import * as THREE from 'three';
import type { TelemetryData } from '../types';
import { Box } from 'lucide-react';

interface DigitalTwin3DProps {
  telemetry: TelemetryData;
}

export const DigitalTwin3D: React.FC<DigitalTwin3DProps> = ({ telemetry }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090e);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(30, 40, 50);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);

    const gridHelper = new THREE.GridHelper(100, 40, 0x00f2fe, 0x1e2638);
    scene.add(gridHelper);

    const buildingMat = new THREE.MeshBasicMaterial({ color: 0x111927, wireframe: true });
    for (let i = 0; i < 8; i++) {
      const boxGeo = new THREE.BoxGeometry(
        Math.random() * 8 + 4,
        Math.random() * 20 + 10,
        Math.random() * 8 + 4
      );
      const mesh = new THREE.Mesh(boxGeo, buildingMat);
      mesh.position.set(
        (Math.random() - 0.5) * 70,
        boxGeo.parameters.height / 2,
        (Math.random() - 0.5) * 70
      );
      scene.add(mesh);
    }

    const carMat = new THREE.MeshBasicMaterial({ color: 0x00ff9d });
    const cars: THREE.Mesh[] = [];
    for (let i = 0; i < 15; i++) {
      const carGeo = new THREE.BoxGeometry(2, 1, 4);
      const carMesh = new THREE.Mesh(carGeo, carMat);
      carMesh.position.set(-40 + i * 6, 0.5, (i % 2 === 0 ? 5 : -5));
      scene.add(carMesh);
      cars.push(carMesh);
    }

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      scene.rotation.y += 0.002;

      cars.forEach((car, idx) => {
        car.position.x += (idx % 2 === 0 ? 0.2 : -0.2);
        if (car.position.x > 40) car.position.x = -40;
        if (car.position.x < -40) car.position.x = 40;
      });

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Box color="var(--accent-cyan)" />
            3D URBAN TRAFFIC DIGITAL TWIN
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Real-time WebGL city simulation updated asynchronously by edge camera telemetry
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <span className="badge badge-green">WebGL Active</span>
          <span className="badge badge-cyan">Three.js Engine</span>
        </div>
      </div>

      <div
        className="glass-panel"
        style={{
          flex: 1,
          minHeight: '450px',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '450px' }} />

        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          background: 'rgba(10, 13, 22, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '12px 18px',
          borderRadius: '8px',
          border: '1px solid var(--border-dim)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          display: 'flex',
          gap: '20px'
        }}>
          <div>PRESSURE: <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{telemetry.pressure.toFixed(2)}</span></div>
          <div>ACTIVE VEHICLES: <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{telemetry.vehicles}</span></div>
          <div>CAMERA NODES: <span style={{ color: 'var(--accent-orange)', fontWeight: 700 }}>8 SYNCED</span></div>
        </div>
      </div>
    </div>
  );
};
