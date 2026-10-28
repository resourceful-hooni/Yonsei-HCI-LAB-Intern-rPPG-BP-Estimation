import { useEffect } from 'react';
import { Navigate, NavLink, Route, Routes, useLocation } from 'react-router-dom';
import MetaTags from './MetaTags';
import { useLang } from './contexts/LangContext';
import MeasurementPage from './pages/MeasurementPage';
import SummaryPage from './pages/SummaryPage';
import LifestyleGuidePage from './pages/LifestyleGuidePage';

/* ---------------------------------------------------------------
   Global ripple effect: attaches a CSS class on pointerdown so
   the ::after pseudo-element triggers the ripple keyframe.
--------------------------------------------------------------- */
function useButtonRipple() {
  useEffect(() => {
    const applyRipple = (target, clientX, clientY) => {
      const el = target.closest('button:not(:disabled), a, .touchable');
      if (!el) return;
      const rect = el.getBoundingClientRect();
      el.style.setProperty('--rx', `${clientX - rect.left}px`);
      el.style.setProperty('--ry', `${clientY - rect.top}px`);
      el.classList.add('rippling');
      const remove = () => el.classList.remove('rippling');
      el.addEventListener('animationend', remove, { once: true });
      // fallback in case animationend never fires
      setTimeout(remove, 600);
    };

    const onPointerDown = (e) => applyRipple(e.target, e.clientX, e.clientY);
    const onTouchStart = (e) => {
      const touch = e.touches && e.touches[0];
      if (!touch) return;
      applyRipple(e.target, touch.clientX, touch.clientY);
    };

    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('touchstart', onTouchStart, { passive: true });
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('touchstart', onTouchStart);
    };
  }, []);
}

/* ---------------------------------------------------------------
   Bottom navigation — only visible on mobile via CSS.
   NavLink adds the "active" class automatically.
--------------------------------------------------------------- */
function BottomNav() {
  // Re-render on route change so active state updates instantly
  useLocation();
  const { t } = useLang();
  return (
    <nav className="bottom-nav" aria-label={t('app_nav_aria')}>
      <NavLink to="/measurement">
        <span className="nav-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
        </span>
        {t('nav_measure')}
      </NavLink>
      <NavLink to="/summary">
        <span className="nav-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
        </span>
        {t('nav_summary')}
      </NavLink>
      <NavLink to="/lifestyle">
        <span className="nav-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </span>
        {t('nav_lifestyle')}
      </NavLink>
    </nav>
  );
}

function App() {
  useButtonRipple();
  const { t, toggle } = useLang();

  return (
    <>
      <MetaTags />
      <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <strong>VisiVital</strong>
            <p>{t('brand_sub')}</p>
          </div>
        </div>
        <nav className="tabs">
          <NavLink to="/measurement">{t('nav_measure')}</NavLink>
          <NavLink to="/summary">{t('nav_summary')}</NavLink>
          <NavLink to="/lifestyle">{t('nav_lifestyle')}</NavLink>
        </nav>
        <button
          className="lang-toggle"
          onClick={toggle}
          aria-label={t('app_lang_aria')}
        >
          {t('lang_toggle')}
        </button>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/measurement" replace />} />
        <Route path="/measurement" element={<MeasurementPage />} />
        <Route path="/summary" element={<SummaryPage />} />
        <Route path="/lifestyle" element={<LifestyleGuidePage />} />
      </Routes>

      <BottomNav />

      <footer className="app-footer">
        <small>
          Developer: Kim Jihoon | Affiliation: Yonsei HCI LAB (Intern) | GitHub: @resourceful-hooni
        </small>
      </footer>
      </div>
    </>
  );
}

export default App;
