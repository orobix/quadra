/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./docs_overrides/home.html"],
  // darkMode: "class",
  theme: {
    fontFamily: {
      sans: ["Roboto", "sans-serif"],
      body: ["Roboto", "sans-serif"],
      mono: ["ui-monospace", "monospace"],
    },
  },
  corePlugins: {
    preflight: false,
  },
  plugins: [],
}

