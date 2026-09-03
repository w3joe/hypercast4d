import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8765', '/health': 'http://127.0.0.1:8765' },
  },
  build: {
    outDir: '../src/hypercast4d/web_dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', '@tanstack/react-query'],
          editor: ['@dnd-kit/core', '@dnd-kit/sortable', '@dnd-kit/utilities'],
          charts: ['recharts'],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
})
