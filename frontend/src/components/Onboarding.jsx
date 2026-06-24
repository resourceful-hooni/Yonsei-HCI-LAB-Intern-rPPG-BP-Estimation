import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLang } from '../contexts/LangContext';

const ONBOARD_KEY = 'visi_onboarded_v1';

export default function Onboarding() {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    try {
      if (!localStorage.getItem(ONBOARD_KEY)) setOpen(true);
    } catch (_) { /* ignore */ }
  }, []);

  const steps = [
    { icon: '📷', title: t('ob_1_title'), body: t('ob_1_body') },
    { icon: '📊', title: t('ob_2_title'), body: t('ob_2_body') },
    { icon: '💡', title: t('ob_3_title'), body: t('ob_3_body') },
  ];

  const finish = () => {
    try { localStorage.setItem(ONBOARD_KEY, '1'); } catch (_) { /* ignore */ }
    setOpen(false);
  };

  if (!open) return null;
  const last = step >= steps.length - 1;
  const s = steps[step];

  return createPortal(
    <div className="onboard-backdrop" role="dialog" aria-modal="true" aria-label={s.title}>
      <div className="onboard-card">
        <div className="onboard-icon" aria-hidden="true">{s.icon}</div>
        <h2 className="onboard-title">{s.title}</h2>
        <p className="onboard-body">{s.body}</p>
        <div className="onboard-dots">
          {steps.map((_, i) => (
            <span key={i} className={`onboard-dot${i === step ? ' active' : ''}`} />
          ))}
        </div>
        <div className="onboard-actions">
          <button className="ghost" onClick={finish}>{t('ob_skip')}</button>
          <button onClick={() => (last ? finish() : setStep(step + 1))}>
            {last ? t('ob_start') : t('ob_next')}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
