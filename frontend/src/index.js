import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { LangProvider } from './contexts/LangContext';
import './styles/globalStyles.css';

const root = createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <LangProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </LangProvider>
  </React.StrictMode>
);
