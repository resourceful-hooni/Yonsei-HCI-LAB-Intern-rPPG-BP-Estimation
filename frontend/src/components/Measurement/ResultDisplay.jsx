import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLang } from '../../contexts/LangContext';
import useCountUp from '../../hooks/useCountUp';

function ResultDisplay({ result, onRetry }) {
  const navigate = useNavigate();
  const { t } = useLang();
  const [shared, setShared] = useState(false);

  const confidencePercent = Math.round((result.confidence || 0) * 100);
  const heartRate = Math.round(Number(result.heart_rate || 0));
  const signalQuality = Math.round(Number(result.signal_quality || 0) * 100);

  // Animated count-up for the headline numbers (modern, satisfying reveal).
  const sys = useCountUp(result.bp_systolic);
  const dia = useCountUp(result.bp_diastolic);
  const glucose = useCountUp(result.blood_sugar);
  const hr = useCountUp(heartRate);

  // Light haptic pulse when the result appears (mobile).
  useEffect(() => {
    if (navigator.vibrate) { try { navigator.vibrate(18); } catch (_) { /* ignore */ } }
  }, []);

  const getGlucoseStatus = (value) => {
    if (value < 90) return { label: t('res_gl_low'), description: t('res_gl_low_desc') };
    if (value <= 110) return { label: t('res_gl_normal'), description: t('res_gl_normal_desc') };
    return { label: t('res_gl_high'), description: t('res_gl_high_desc') };
  };
  const glucoseStatus = getGlucoseStatus(Number(result.blood_sugar || 0));

  const handleShare = async () => {
    const text =
      `VisiVital\n` +
      `${t('res_bp_label')}: ${result.bp_systolic}/${result.bp_diastolic} ${t('res_bp_unit') || 'mmHg'}\n` +
      `${t('res_glucose_label')}: ${result.blood_sugar} ${t('res_glucose_unit') || 'mg/dL'}\n` +
      (heartRate ? `${t('res_hr_label')}: ${heartRate} ${t('res_hr_unit')}\n` : '') +
      `${t('res_confidence')}: ${confidencePercent}%`;
    try {
      if (navigator.share) {
        await navigator.share({ title: 'VisiVital', text });
        return;
      }
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text);
        setShared(true);
        setTimeout(() => setShared(false), 1800);
      }
    } catch (_) { /* user cancelled / unsupported */ }
  };

  return (
    <div className="card">
      <h2>{t('res_title')}</h2>
      <p className="subtitle" style={{ marginBottom: 6 }}>{t('res_bp_label')}</p>
      <div className="result-number">{sys} / {dia} <small>{t('res_bp_unit')}</small></div>
      <p className="bp-sublabels"><span>{t('res_bp_systolic')}</span><span>/</span><span>{t('res_bp_diastolic')}</span></p>
      <p className="subtitle" style={{ marginBottom: 6 }}>{t('res_glucose_label')}</p>
      <div className="result-number">{glucose} <small>{t('res_glucose_unit')}</small></div>

      {/* Measurement metrics from the improved rPPG pipeline — visible at a glance. */}
      {(heartRate > 0 || signalQuality > 0) && (
        <div style={{ display: 'flex', gap: 10, marginTop: 14, marginBottom: 4 }}>
          {heartRate > 0 && (
            <div style={{ flex: 1, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 12, padding: '10px 12px' }}>
              <div style={{ opacity: 0.6, fontSize: '0.78rem' }}>{t('res_hr_label')}</div>
              <div style={{ fontSize: '1.45rem', fontWeight: 700, lineHeight: 1.2 }}>
                {hr} <small style={{ fontSize: '0.78rem', fontWeight: 500 }}>{t('res_hr_unit')}</small>
              </div>
            </div>
          )}
          {signalQuality > 0 && (
            <div style={{ flex: 1, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 12, padding: '10px 12px' }}>
              <div style={{ opacity: 0.6, fontSize: '0.78rem' }}>{t('res_signal_quality')}</div>
              <div style={{ fontSize: '1.45rem', fontWeight: 700, lineHeight: 1.2 }}>
                {signalQuality}<small style={{ fontSize: '0.9rem', fontWeight: 500 }}>%</small>
              </div>
              <div style={{ height: 6, background: 'var(--border)', borderRadius: 6, marginTop: 6, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${signalQuality}%`, background: 'linear-gradient(90deg, var(--primary), var(--secondary))' }} />
              </div>
            </div>
          )}
        </div>
      )}

      <p>{t('res_confidence')}: {confidencePercent}%</p>
      <p className="subtitle">{t('res_conf_sub')}</p>

      {result.flags?.low_quality && (
        <p className="quality-warning" role="alert" style={{ marginTop: 8 }}>
          {t('res_low_quality')}
        </p>
      )}

      <p className="subtitle">
        {t('res_bp_source')}: {result.bp_source === 'research_model' ? t('res_bp_src_main') : t('res_bp_src_fallback')}
      </p>
      <div style={{ marginTop: 8, marginBottom: 8 }}>
        <img
          src="/images/ms_tcn_predictions.png"
          alt={t('res_img_alt')}
          style={{ width: '100%', borderRadius: 10, border: '1px solid #e2e8f0' }}
          onError={(e) => { e.currentTarget.style.display = 'none'; }}
        />
        <small style={{ display: 'block', textAlign: 'center', marginTop: 4, opacity: 0.6 }}>{t('res_img_caption')}</small>
      </div>
      <p className="subtitle">{t('res_glucose_source')}</p>

      <div className="card" style={{ marginTop: 12, background: 'var(--surface-2)' }}>
        <h3 style={{ marginTop: 0 }}>{t('res_glucose_title')}</h3>
        <p><strong>{glucoseStatus.label}</strong></p>
        <p style={{ marginBottom: 0 }}>{glucoseStatus.description}</p>
      </div>

      <div className="actions">
        <button onClick={() => navigate('/summary')}>{t('res_btn_summary')}</button>
        <button className="ghost" onClick={handleShare}>{shared ? t('res_share_done') : t('res_share')}</button>
        <button className="ghost" onClick={onRetry}>{t('res_btn_retry')}</button>
      </div>

      <p className="disclaimer" style={{ marginTop: 16 }}>
        {t('res_disclaimer')}
      </p>
    </div>
  );
}

export default ResultDisplay;
