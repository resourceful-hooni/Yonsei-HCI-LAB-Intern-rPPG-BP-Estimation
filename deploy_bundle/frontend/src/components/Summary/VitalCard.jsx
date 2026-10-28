import { useState } from 'react';
import { createPortal } from 'react-dom';
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { useLang } from '../../contexts/LangContext';

const buildInfoContent = (t) => ({
  'blood-pressure': {
    title: t('vc_bp_info_title'),
    unit: t('vc_bp_unit'),
    body: [
      { range: t('vc_bp_normal'), value: t('vc_bp_normal_val'), color: '#22c55e' },
      { range: t('vc_bp_elevated'), value: t('vc_bp_elevated_val'), color: '#eab308' },
      { range: t('vc_bp_stage1'), value: t('vc_bp_stage1_val'), color: '#f97316' },
      { range: t('vc_bp_stage2'), value: t('vc_bp_stage2_val'), color: '#ef4444' },
    ],
    note: t('vc_bp_note'),
  },
  'blood-sugar': {
    title: t('vc_gl_info_title'),
    unit: t('vc_gl_unit'),
    body: [
      { range: t('vc_gl_normal'), value: t('vc_gl_normal_val'), color: '#22c55e' },
      { range: t('vc_gl_ifg'), value: t('vc_gl_ifg_val'), color: '#eab308' },
      { range: t('vc_gl_dm'), value: t('vc_gl_dm_val'), color: '#ef4444' },
    ],
    note: t('vc_gl_note'),
  },
});

function VitalCard({ type, systolic, diastolic, value, status, trendData = [] }) {
  const [showInfo, setShowInfo] = useState(false);
  const { t } = useLang();
  const isBP = type === 'blood-pressure';
  const title = isBP ? t('vc_bp_title') : t('vc_glucose_title');
  const display = isBP ? `${systolic} / ${diastolic}` : `${value}`;
  const statusText = String(status || '').toLowerCase();
  const tone = statusText.includes('안정') || statusText.includes('stable')
    ? 'stable'
    : (statusText.includes('관심') || statusText.includes('watch') || statusText.includes('주의') || statusText.includes('attention'))
      ? 'watch'
      : 'care';
  const values = (trendData || []).map((d) => Number(d.value || 0));
  const localMin = values.length ? Math.min(...values) : 0;
  const localMax = values.length ? Math.max(...values) : 1;
  const yMin = Math.max(0, localMin - Math.max(2, (localMax - localMin) * 0.2));
  const yMax = localMax + Math.max(2, (localMax - localMin) * 0.2);
  const infoContent = buildInfoContent(t);
  const info = infoContent[type] || infoContent['blood-pressure'];

  return (
    <div className={`card vital-card touchable ${tone}`}>
      <div className="row between">
        <h3>{title}</h3>
        <span className="status-badge">{status}</span>
      </div>
      <div className="legend-row" style={{ marginBottom: 6 }}>
        <span className="legend-item">
          <span className={`legend-swatch ${isBP ? 'bp' : 'glucose'}`} />{t('vc_trend_line')}
        </span>
        <span className={`legend-pill ${tone}`}>{t('vc_bg_status')}: {status}</span>
      </div>
      <div className="result-number">{display}</div>
      {isBP && (
        <p className="bp-sublabels"><span>{t('vc_bp_systolic')}</span><span>/</span><span>{t('vc_bp_diastolic')}</span></p>
      )}
      <div style={{ width: '100%', height: 64, marginTop: 4 }}>
        <ResponsiveContainer>
          <LineChart data={trendData}>
            <YAxis hide domain={[yMin, yMax]} />
            <Tooltip formatter={(v) => Number(v).toFixed(1)} labelFormatter={(l) => `${l}`} />
            <Line type="monotone" dataKey="value" stroke={isBP ? '#5C7CFA' : '#9775FA'} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <button className="ghost" onClick={(e) => { e.stopPropagation(); setShowInfo(true); }}>{t('vc_info_btn')}</button>

      {showInfo && createPortal(
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={info.title}
          onClick={(e) => { if (e.target === e.currentTarget) setShowInfo(false); }}
        >
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row between" style={{ marginBottom: 12 }}>
              <h3 style={{ margin: 0 }}>{info.title}</h3>
              <button
                className="ghost"
                onClick={() => setShowInfo(false)}
                style={{ padding: '4px 10px', fontSize: '1rem' }}
                aria-label={t('vc_modal_close')}
              >✕</button>
            </div>
            <small style={{ display: 'block', marginBottom: 12, opacity: 0.7 }}>{t('vc_modal_unit')}: {info.unit}</small>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', marginBottom: 14 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', paddingBottom: 6, opacity: 0.6, fontWeight: 500 }}>{t('vc_modal_range')}</th>
                  <th style={{ textAlign: 'left', paddingBottom: 6, opacity: 0.6, fontWeight: 500 }}>{t('vc_modal_criteria')}</th>
                </tr>
              </thead>
              <tbody>
                {info.body.map((row) => (
                  <tr key={row.range}>
                    <td style={{ padding: '5px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: row.color, display: 'inline-block', flexShrink: 0 }} />
                      {row.range}
                    </td>
                    <td style={{ padding: '5px 0', paddingLeft: 8, color: row.color, fontWeight: 600 }}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="disclaimer" style={{ margin: 0, fontSize: '0.8rem' }}>{info.note}</p>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

export default VitalCard;
