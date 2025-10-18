import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    open: true,
    proxy: {
      // Proxy model asset requests to Hugging Face to avoid CORS and SPA fallback
      '/Xenova': {
        target: 'https://huggingface.co',
        changeOrigin: true,
        secure: true,
      },
    },
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'three',
      '@react-three/fiber',
      '@react-three/drei',
      'ml-pca',
    ],
  },
});
