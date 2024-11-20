// index.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Ensure the path is correct without the .jsx extension
import './index.css';

import { ThemeProvider } from "@material-tailwind/react";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
