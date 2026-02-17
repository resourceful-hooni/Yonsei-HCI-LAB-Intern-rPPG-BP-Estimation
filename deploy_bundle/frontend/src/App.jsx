import { Navigate, NavLink, Route, Routes } from 'react-router-dom';
import MeasurementPage from './pages/MeasurementPage';
import SummaryPage from './pages/SummaryPage';
import LifestyleGuidePage from './pages/LifestyleGuidePage';

function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <strong>VisiVital</strong>
            <p>rPPG 기반 건강 참고 모니터링</p>
          </div>
        </div>
        <nav className="tabs">
          <NavLink to="/measurement">측정</NavLink>
          <NavLink to="/summary">요약</NavLink>
          <NavLink to="/lifestyle">생활관리</NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/measurement" replace />} />
        <Route path="/measurement" element={<MeasurementPage />} />
        <Route path="/summary" element={<SummaryPage />} />
        <Route path="/lifestyle" element={<LifestyleGuidePage />} />
      </Routes>

      <footer className="app-footer">
        <small>
          Developer: Kim Jihoon | Affiliation: Yonsei HCI LAB (Intern) | GitHub: @resourceful-hooni
        </small>
      </footer>
    </div>
  );
}

export default App;
