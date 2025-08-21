import tailwindcss from '@tailwindcss/vite'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  app: {
    head: {
      title: 'Détecteur de Pneumonie par Rayons X - IA KNN & CNN',
      meta: [
        {
          name: 'description',
          content:
              "Testez gratuitement notre détecteur de pneumonie basé sur l'intelligence artificielle (KNN & CNN). Uploadez une radiographie pulmonaire ou utilisez des exemples pour obtenir une prédiction immédiate avec probabilités détaillées."
        },
        {
          name: 'keywords',
          content:
              "pneumonie, détection pneumonie, rayons X, radiographie, intelligence artificielle, IA médicale, knn, cnn, machine learning santé"
        },
        { property: 'og:title', content: 'Détecteur de Pneumonie par Rayons X - IA KNN & CNN' },
        {
          property: 'og:description',
          content:
              "Analysez vos radios pulmonaires avec notre détecteur de pneumonie basé sur l'IA (KNN et CNN). Résultats immédiats et probabilités détaillées."
        },
        { property: 'og:type', content: 'website' },
        { property: 'og:url', content: 'https://ai-pneumonia-detector.dibodev.fr' }
      ],
      script: [
        {
          src: 'https://umami.dibodev.fr/script.js',
          defer: true,
          'data-website-id': '92fca472-e013-4d60-8cdd-43145bfbcf78',
        },
      ],
    },
  },
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  ssr: true,
  css: ['~/assets/css/main.css'],
  vite: {
    plugins: [tailwindcss()],
  },
  nitro: {
    preset: 'static',
  },
  modules: ['@nuxtjs/sitemap', '@nuxtjs/robots', '@nuxtjs/google-fonts'],
  site: {
    url: 'https://ai-pneumonia-detector.dibodev.fr',
    name: 'Détecteur de Pneumonie par Rayons X - IA KNN & CNN',
  },
  googleFonts: {
    families: {
      Rubik: {
        wght: [400, 500, 600, 700], // Use only the weights you need (e.g., Regular, Medium, Bold)
      },
    },
    display: 'swap', // Ensures text is visible during font loading
    subsets: ['latin'], // Use 'latin-ext' if you need extended Latin characters
    download: true, // Download fonts locally
    base64: true, // Encode fonts in Base64 to avoid external requests
    inject: true, // Inject the generated CSS into the project
    overwriting: true, // Overwrite existing font files to avoid duplicates
    outputDir: 'assets/fonts', // Store downloaded fonts in assets/fonts
    stylePath: 'assets/css/google-fonts.css', // Path for the generated CSS
    fontsDir: 'fonts', // Relative to outputDir
    fontsPath: '../fonts', // Path used in the CSS file
    prefetch: false, // Disable prefetch for SSG (not needed with local fonts)
    preconnect: false, // Disable preconnect for SSG (not needed with local fonts)
    preload: true, // Preload the font CSS for faster rendering
    useStylesheet: false, // Use inline CSS (via base64) instead of external stylesheet
  },
  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.API_BASE_URL || 'https://api.dibodev.fr/pneumonia-detector',
    },
  },
})
