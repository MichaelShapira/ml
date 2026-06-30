/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b0b0f",
        panel: "#14141b",
        edge: "#23232b",
        accent: "#e8590c",
      },
    },
  },
  plugins: [],
};
