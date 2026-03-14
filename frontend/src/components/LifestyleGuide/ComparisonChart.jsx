import { useLang } from '../../contexts/LangContext';

function ComparisonChart({ title, percentile, color }) {
  const { t } = useLang();
  return (
    <div className="card comparison-card">
      <h3>{title} {t('lg_comp_suffix')}</h3>
      <div className="bar-wrap">
        <div className="bar low">{t('lg_comp_low')}</div>
        <div className="bar normal">{t('lg_comp_normal')}</div>
        <div className="bar high">{t('lg_comp_high')}</div>
        <div className="marker" style={{ left: `${percentile}%`, borderColor: color }} />
      </div>
      <p className="percentile">{t('lg_comp_my_pos')}: <strong>{percentile}%</strong></p>
    </div>
  );
}

export default ComparisonChart;
