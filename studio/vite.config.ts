import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
      '/telemetry': 'http://127.0.0.1:8080',
      '/cameras': 'http://127.0.0.1:8080',
      '/analytics': 'http://127.0.0.1:8080',
      '/alerts': 'http://127.0.0.1:8080',
      '/logs': 'http://127.0.0.1:8080',
      '/settings': 'http://127.0.0.1:8080',
      '/notifications': 'http://127.0.0.1:8080',
      '/ws': {
        target: 'ws://127.0.0.1:8080',
        ws: true
      }
    }
  }
})
