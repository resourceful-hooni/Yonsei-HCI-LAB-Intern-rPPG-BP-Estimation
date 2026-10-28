import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  // CRA 시절 .env 변수(REACT_APP_*)를 그대로 사용
  envPrefix: 'REACT_APP_',
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:5000'
    }
  },
  preview: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:5000'
    }
  },
  build: {
    outDir: 'build'
  }
});
