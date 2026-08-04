import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/health': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/telemetry': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/cameras': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/analytics': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/alerts': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/logs': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/settings': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/notifications': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://127.0.0.1:8080',
        ws: true
      }
    }
  }
})
