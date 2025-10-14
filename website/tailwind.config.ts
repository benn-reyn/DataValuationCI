import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './pages/**/*.{ts,tsx}',
    './content/**/*.{ts,tsx,json}'
  ],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: '#F56600', dark: '#C24C00' } // Clemson-ish
      },
      boxShadow: {
        soft: '0 8px 30px rgba(0,0,0,0.08)'
      }
    },
    container: { center: true, padding: '1rem' }
  },
  plugins: []
}
export default config
