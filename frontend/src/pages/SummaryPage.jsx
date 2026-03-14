import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import VitalCard from '../components/Summary/VitalCard';
import { fetchDailySummary, fetchTrends } from '../services/apiService';
import { useLang } from '../contexts/LangContext';

function SummaryPage() {
  const { t } = useLang();
  const [daily, setDaily] = useState(null);
  const [trends, setTrends] = useState({ bp_trend: [], glucose_trend: [], trend_percentages: {} });
  const [error, setError] = useState('');
  const [loaded, setLoaded] = useState(false);
  const navigate = useNavigate();


  useEffect(() => {
    Promise.allSettled([fetchDailySummary('demo-user'), fetchTrends('demo-user', 7)])
      .then(([dailyRes, trendRes]) => {
        if (dailyRes.status === 'fulfilled') {
          setDaily(dailyRes.value.data);
        }
        if (trendRes.status === 'fulfilled') {
          setTrends(trendRes.value.data || {});
        }
        if (dailyRes.status === 'rejected') {
          setError(dailyRes.reason?.message || t('sp_loading'));
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoaded(true));
  }, []);

  if (error) return <div className="page"><p>{error}</p></div>;
  if (!loaded) return <div className="page"><p>{t('sp_loading')}</p></div>;
  if (!daily) {
    return (
      <div className="page">
        <h1>{t('sp_title')}</h1>
        <p className="subtitle">{t('sp_no_data_sub')}</p>
        <div className="card">
          <h3>{t('sp_guide_title')}</h3>
          <p>{t('sp_guide_body')}</p>
          <button className="ghost" onClick={() => navigate('/measurement')}>{t('sp_go_measure')}</button>
        </div>
      </div>
    );
  }

  const confidenceTrend = (daily.confidence_trend && daily.confidence_trend.length)
    ? daily.confidence_trend
    : [{ date: 'today', value: Number(daily.current_values?.confidence || 0) }];

  const confVals = confidenceTrend.map((d) => Number(d.value || 0));
  const confMin = confVals.length > 1 ? Math.min(...confVals) : null;
  const confMax = confVals.length > 1 ? Math.max(...confVals) : null;

  const parseTs = (timestamp) => {
    const raw = String(timestamp || '').trim();
    if (!raw) return null;

    // Case 1: explicit timezone included (Z or ±HH:MM)
    if (/[zZ]|[+\-]\d{2}:?\d{2}$/.test(raw)) {
      const direct = new Date(raw);
      return Number.isNaN(direct.getTime()) ? null : direct;
    }

    // Case 2: timezone-naive string -> interpret as Asia/Seoul local time
    // Supported examples: "2026-02-22 16:38:00", "2026-02-22T16:38:00"
    const normalized = raw.replace('T', ' ');
    const match = normalized.match(/^(\d{4})-(\d{2})-(\d{2})\s(\d{2}):(\d{2})(?::(\d{2}))?/);
    if (match) {
      const year = Number(match[1]);
      const month = Number(match[2]);
      const day = Number(match[3]);
      const hour = Number(match[4]);
      const minute = Number(match[5]);
      const second = Number(match[6] || 0);

      // Convert Seoul local clock to UTC epoch (KST = UTC+9)
      const utcMs = Date.UTC(year, month - 1, day, hour - 9, minute, second);
      const date = new Date(utcMs);
      return Number.isNaN(date.getTime()) ? null : date;
    }

    // Fallback
    const fallback = new Date(raw);
    return Number.isNaN(fallback.getTime()) ? null : fallback;
  };

  const formatTs = (timestamp, withDate = true) => {
    const date = parseTs(timestamp);
    if (!date) return '-';
    const parts = new Intl.DateTimeFormat('ko-KR', {
      timeZone: 'Asia/Seoul',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).formatToParts(date);
    const find = (type) => parts.find((p) => p.type === type)?.value || '';
    const mm = find('month');
    const dd = find('day');
    const hh = find('hour');
    const mi = find('minute');
    return withDate ? `${mm}-${dd} ${hh}:${mi}` : `${hh}:${mi}`;
  };

  const combinedTrendDataRaw = (trends.combined_trend && trends.combined_trend.length)
    ? trends.combined_trend.map((d) => ({
      label: formatTs(d.timestamp, true),
      timeLabel: formatTs(d.timestamp, false),
      timestamp: d.timestamp,
      tsMs: parseTs(d.timestamp)?.getTime() ?? Number.MAX_SAFE_INTEGER,
      bp: Number(d.bp || 0),
      glucose: Number(d.glucose || 0),
    }))
    : (trends.bp_trend || []).map((bpItem, idx) => ({
      label: formatTs(bpItem.timestamp, true),
      timeLabel: formatTs(bpItem.timestamp, false),
      timestamp: bpItem.timestamp,
      tsMs: parseTs(bpItem.timestamp)?.getTime() ?? Number.MAX_SAFE_INTEGER,
      bp: Number(bpItem.value || 0),
      glucose: Number((trends.glucose_trend || [])[idx]?.value || 0),
    }));

  const combinedTrendData = [...combinedTrendDataRaw].sort((a, b) => a.tsMs - b.tsMs);

  const latestTimeLabel = combinedTrendData.length
    ? (combinedTrendData[combinedTrendData.length - 1].label || formatTs(combinedTrendData[combinedTrendData.length - 1].timestamp, true) || '-')
    : '-';

  const qualityChecklist = (daily.quality_checklist && daily.quality_checklist.length)
    ? daily.quality_checklist
    : [
      {
        id: 'lighting',
        label: t('sp_q_lighting'),
        status: Number(daily.current_values?.confidence || 0) >= 0.75 ? '좋음' : '보통',
        tip: t('cam_fail_tip3'),
      },
      {
        id: 'movement',
        label: t('sp_q_movement'),
        status: '보통',
        tip: t('cam_fail_tip2'),
      },
      {
        id: 'alignment',
        label: t('sp_q_alignment'),
        status: Number(daily.current_values?.confidence || 0) >= 0.75 ? '좋음' : '개선 필요',
        tip: t('cam_fail_tip1'),
      },
    ];
  const qualitySourceText = daily.quality_meta?.is_measured_directly
    ? t('sp_qs_direct')
    : (daily.quality_checklist && daily.quality_checklist.length)
      ? t('sp_qs_rule')
      : t('sp_qs_fallback');

  const normalizeStatus = (value) => {
    const v = String(value || '').toLowerCase();
    if (v.includes('좋') || v.includes('good')) return 'good';
    if (v.includes('보통') || v.includes('fair') || v.includes('mid')) return 'mid';
    return 'bad';
  };

  const statusText = (value) => {
    const code = normalizeStatus(value);
    if (code === 'good') return t('sp_q_good');
    if (code === 'mid') return t('sp_q_mid');
    return t('sp_q_bad');
  };

  const localizeStatusLabel = (value) => {
    const v = String(value || '').toLowerCase();
    if (v.includes('안정') || v.includes('stable')) return t('st_stable');
    if (v.includes('관심') || v.includes('watch') || v.includes('주의') || v.includes('attention')) return t('st_watch');
    return t('st_care');
  };

  const localizedStatusLabel = localizeStatusLabel(daily?.status?.status_label);

  return (
    <div className="page">
      <h1>{t('sp_title')}</h1>
      <p className="subtitle">{t('sp_subtitle')}</p>
      <div className="status-badge level">{localizedStatusLabel}</div>
      {(() => {
        const bpTrend = trends.trend_percentages?.bp || 0;
        const glTrend = trends.trend_percentages?.glucose || 0;
        const warnings = [];
        if (bpTrend > 5) warnings.push(t('sp_bp_warn_up', { v: bpTrend.toFixed(1) }));
        if (bpTrend < -5) warnings.push(t('sp_bp_warn_down', { v: Math.abs(bpTrend).toFixed(1) }));
        if (glTrend > 5) warnings.push(t('sp_gl_warn_up', { v: glTrend.toFixed(1) }));
        if (warnings.length === 0) return null;
        return (
          <div className="trend-warning" role="alert">
            {warnings.map((w, i) => <span key={i}>{w}</span>)}
            <small>{t('sp_warn_note')}</small>
          </div>
        );
      })()}
      <p>{daily.summary_text}</p>

      <div className="grid">
        <VitalCard
          type="blood-pressure"
          systolic={daily.current_values.bp_systolic}
          diastolic={daily.current_values.bp_diastolic}
          status={localizedStatusLabel}
          trendData={(trends.bp_trend || []).map((d) => ({ value: Number(d.value || 0), date: d.date }))}
        />

        <VitalCard
          type="blood-sugar"
          value={daily.current_values.blood_sugar}
          status={localizedStatusLabel}
          trendData={(trends.glucose_trend || []).map((d) => ({ value: Number(d.value || 0), date: d.date }))}
        />

        <div className="card confidence-card">
          <h3>{t('sp_confidence_title')}</h3>
          <div className="legend-row" style={{ marginBottom: 6 }}>
            <span className="legend-item"><span className="legend-swatch confidence" />{t('sp_legend_conf')}</span>
            <span className="legend-item"><span className="legend-swatch confidence-bg" />{t('sp_legend_conf_bg')}</span>
          </div>
          <div style={{ width: '100%', height: 100 }}>
            <ResponsiveContainer>
              <LineChart data={confidenceTrend} margin={{ left: 0, right: 8 }}>
                <YAxis
                  domain={[0, 1]}
                  width={36}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  tick={{ fontSize: 10 }}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                />
                <Tooltip formatter={(v) => `${Math.round(Number(v || 0) * 100)}%`} labelFormatter={(l) => `${l}`} />
                <ReferenceLine y={0.75} stroke="#22c55e" strokeDasharray="4 3" label={{ value: t('sp_conf_best'), position: 'insideTopRight', fontSize: 10, fill: '#22c55e' }} />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#5C7CFA"
                  dot={(props) => {
                    const { cx, cy, value: v, index } = props;
                    const isMin = confMin !== null && v === confMin;
                    const isMax = confMax !== null && v === confMax;
                    if (!isMin && !isMax) return <circle key={index} cx={cx} cy={cy} r={2} fill="#5C7CFA" />;
                    return (
                      <g key={index}>
                        <circle cx={cx} cy={cy} r={5} fill={isMax ? '#22c55e' : '#f97316'} stroke="#fff" strokeWidth={1.5} />
                        <text x={cx} y={cy - 9} textAnchor="middle" fontSize={9} fill={isMax ? '#22c55e' : '#f97316'}>
                          {isMax ? t('sp_conf_best') : t('sp_conf_worst')}
                        </text>
                      </g>
                    );
                  }}
                  strokeWidth={2}
                /></LineChart>
            </ResponsiveContainer>
          </div>
          <small>
            {t('sp_conf_avg_prefix')} {Math.round((confidenceTrend.reduce((a, c) => a + (c.value || 0), 0) /
              Math.max(1, confidenceTrend.length)) * 100)}%
          </small>
        </div>

        <div className="card">
          <h3>{t('sp_chart_title')}</h3>
          <div className="row between">
            <div className="row gap">
              <span className={(trends.trend_percentages?.bp || 0) >= 0 ? 'trend up' : 'trend down'}>
                {t('sp_bp_label')} {(trends.trend_percentages?.bp || 0) >= 0 ? '▲' : '▼'} {Math.abs(trends.trend_percentages?.bp || 0).toFixed(1)}%
              </span>
              <span className={(trends.trend_percentages?.glucose || 0) >= 0 ? 'trend up' : 'trend down'}>
                {t('sp_glucose_label')} {(trends.trend_percentages?.glucose || 0) >= 0 ? '▲' : '▼'} {Math.abs(trends.trend_percentages?.glucose || 0).toFixed(1)}%
              </span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={combinedTrendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="label"
                minTickGap={24}
                interval="preserveStartEnd"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => {
                  const s = String(v || '');
                  return s.includes(' ') ? s.split(' ')[1] : s;
                }}
              />
              <YAxis
                yAxisId="left"
                domain={[90, 180]}
                label={{ value: 'mmHg', angle: -90, position: 'insideLeft', offset: 8, style: { fontSize: 10, fill: '#64748b' } }}
                width={46}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[60, 180]}
                label={{ value: 'mg/dL', angle: 90, position: 'insideRight', offset: 8, style: { fontSize: 10, fill: '#64748b' } }}
                width={46}
              />
              <Tooltip
                labelFormatter={(_, payload) => {
                  const row = payload && payload[0] ? payload[0].payload : null;
                  return row?.label || formatTs(row?.timestamp, true) || '-';
                }}
              />
              <Legend />
              <ReferenceLine yAxisId="left" y={120} stroke="#5C7CFA" strokeOpacity={0.6} strokeDasharray="4 3" label={{ value: t('sp_ref_bp_normal'), position: 'insideTopLeft', fontSize: 9, fill: '#5C7CFA' }} />
              <ReferenceLine yAxisId="right" y={110} stroke="#9775FA" strokeOpacity={0.6} strokeDasharray="4 3" label={{ value: t('sp_ref_gl_caution'), position: 'insideTopRight', fontSize: 9, fill: '#9775FA' }} />
              <Line yAxisId="left" type="monotone" dataKey="bp" name={t('sp_bp_label')} stroke="#5C7CFA" strokeWidth={2} dot={{ r: 2 }} />
              <Line yAxisId="right" type="monotone" dataKey="glucose" name={t('sp_glucose_label')} stroke="#9775FA" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
          <small>{t('sp_latest_time')}: {latestTimeLabel}</small>
        </div>

        <div className="card">
          <h3>{t('sp_quality_title')}</h3>
          <small style={{ display: 'block', marginBottom: 8, opacity: 0.8 }}>
            {qualitySourceText}
          </small>
          {qualityChecklist.map((item) => (
            <div key={item.id} className="check-item">
              <div className="row between">
                <strong>{item.label || (item.id === 'lighting' ? t('sp_q_lighting') : item.id === 'movement' ? t('sp_q_movement') : t('sp_q_alignment'))}</strong>
                <span className={`quality ${normalizeStatus(item.status)}`}>{statusText(item.status)}</span>
              </div>
              {Number.isFinite(Number(item.score)) && (
                <small>{t('sp_score_label')}: {Math.round(Number(item.score) * 100)}%</small>
              )}
              <small>{item.tip}</small>
            </div>
          ))}
        </div>
      </div>

      <button onClick={() => navigate('/lifestyle')}>{t('sp_btn_lifestyle')}</button>
    </div>
  );
}

export default SummaryPage;
