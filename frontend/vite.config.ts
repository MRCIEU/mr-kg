import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': '/src'
    }
  },
  server: {
    host: true, // Listen on all addresses including Docker
    port: 5173,
    watch: {
      usePolling: true, // Enable polling for file changes in Docker
    }
  },
  preview: {
    host: true,
    port: 4173
  }
})
