import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative asset paths so the build can be served under /inbox/
export default defineConfig({
  plugins: [react()],
  base: './',
})
