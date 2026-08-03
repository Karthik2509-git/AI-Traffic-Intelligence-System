import React from 'react';
import type { PluginItem } from '../types';
import { Store, Download, Check } from 'lucide-react';

export const PluginStore: React.FC = () => {
  const [plugins, setPlugins] = React.useState<PluginItem[]>([
    {
      id: 'traffic-pro',
      name: 'Intelligent Traffic Suite',
      category: 'Traffic & Urban',
      description: 'Vehicle counting, queue pressure calculation, lane occupancy heatmaps, and NEMA signal control.',
      icon: 'car',
      version: 'v3.1.0',
      installed: true,
      downloads: '14.2k',
      author: 'ATOS Core Team'
    },
    {
      id: 'retail-flow',
      name: 'Retail & Customer Density',
      category: 'Retail & Commercial',
      description: 'Customer foot traffic tracking, queue wait time analysis, store heatmaps, and dwell time stats.',
      icon: 'shop',
      version: 'v1.4.2',
      installed: false,
      downloads: '8.7k',
      author: 'Vision Analytics Labs'
    },
    {
      id: 'ppe-safety',
      name: 'Industrial PPE & Safety Guard',
      category: 'Industrial Safety',
      description: 'Hard hat compliance, hi-vis vest detection, hazardous zone breach alerts, and fall detection.',
      icon: 'safety',
      version: 'v2.0.1',
      installed: true,
      downloads: '11.5k',
      author: 'SafeWork AI'
    },
    {
      id: 'agri-vision',
      name: 'Agriculture & Crop Health',
      category: 'Agriculture & Farm',
      description: 'Livestock counting, crop growth stage tracking, perimeter intruder deterrence, and harvest estimation.',
      icon: 'farm',
      version: 'v1.1.0',
      installed: false,
      downloads: '4.3k',
      author: 'AgriTech AI'
    },
    {
      id: 'campus-security',
      name: 'Smart Campus & ANPR OCR',
      category: 'Security & Access',
      description: 'Automatic License Plate Recognition (ANPR), tailgating prevention, and blacklisted vehicle alert.',
      icon: 'lock',
      version: 'v2.3.0',
      installed: false,
      downloads: '9.8k',
      author: 'SecureCampus'
    }
  ]);

  const toggleInstall = (id: string) => {
    setPlugins((prev) =>
      prev.map((p) => (p.id === id ? { ...p, installed: !p.installed } : p))
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Store color="var(--accent-cyan)" />
            ATOS PLUGIN MARKETPLACE
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Installable domain-specific visual intelligence modules extending camera sensors
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <span className="badge badge-cyan">5 MODULES INSTALLED</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {plugins.map((plugin) => (
          <div key={plugin.id} className="glass-panel-interactive" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="badge badge-purple">{plugin.category}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{plugin.version}</span>
              </div>

              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#fff', marginBottom: '6px' }}>
                {plugin.name}
              </h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: '16px' }}>
                {plugin.description}
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-dim)', paddingTop: '14px', marginTop: '14px' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                By <strong>{plugin.author}</strong> ({plugin.downloads} installs)
              </span>

              <button
                onClick={() => toggleInstall(plugin.id)}
                className={plugin.installed ? 'btn-secondary' : 'btn-primary'}
                style={{ fontSize: '0.78rem', padding: '6px 12px' }}
              >
                {plugin.installed ? (
                  <>
                    <Check size={14} color="var(--accent-green)" /> Installed
                  </>
                ) : (
                  <>
                    <Download size={14} /> Install Plugin
                  </>
                )}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
