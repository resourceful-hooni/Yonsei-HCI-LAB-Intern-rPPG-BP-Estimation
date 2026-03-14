import { useNavigate } from 'react-router-dom';
import { useLang } from '../../contexts/LangContext';

function ResultDisplay({ result, onRetry }) {
  const navigate = useNavigate();
  const { t } = useLang();

  const confidencePercent = Math.round((result.confidence || 0) * 100);

  const getGlucoseStatus = (value) => {
    if (value < 90) {
      return {
        label: t('res_gl_low'),
        description: t('res_gl_low_desc')
      };
    }
    if (value <= 110) {
      return {
        label: t('res_gl_normal'),
        description: t('res_gl_normal_desc')
      };
    }
    return {
      label: t('res_gl_high'),
      description: t('res_gl_high_desc')
    };
  };

  const glucoseStatus = getGlucoseStatus(Number(result.blood_sugar || 0));

  return (
    <div className="card">
      <h2>{t('res_title')}</h2>
      <p className="subtitle" style={{ marginBottom: 6 }}>{t('res_bp_label')}</p>
      <div className="result-number">{result.bp_systolic} / {result.bp_diastolic} <small>{t('res_bp_unit')}</small></div>
      <p className="bp-sublabels"><span>{t('res_bp_systolic')}</span><span>/</span><span>{t('res_bp_diastolic')}</span></p>
      <p className="subtitle" style={{ marginBottom: 6 }}>{t('res_glucose_label')}</p>
      <div className="result-number">{result.blood_sugar} <small>{t('res_glucose_unit')}</small></div>

      <p>{t('res_confidence')}: {confidencePercent}%</p>
      <p className="subtitle">{t('res_conf_sub')}</p>

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

      <div className="card" style={{ marginTop: 12, background: '#fafbff' }}>
        <h3 style={{ marginTop: 0 }}>{t('res_glucose_title')}</h3>
        <p><strong>{glucoseStatus.label}</strong></p>
        <p style={{ marginBottom: 0 }}>{glucoseStatus.description}</p>
      </div>

      <div className="actions">
        <button onClick={() => navigate('/summary')}>{t('res_btn_summary')}</button>
        <button className="ghost" onClick={onRetry}>{t('res_btn_retry')}</button>
      </div>

      <p className="disclaimer" style={{ marginTop: 16 }}>
        {t('res_disclaimer')}
      </p>
    </div>
  );
}

export default ResultDisplay;
