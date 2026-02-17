import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import VitalCard from '../components/Summary/VitalCard';
import { fetchDailySummary, fetchTrends } from '../services/apiService';

function SummaryPage() {
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
          setError(dailyRes.reason?.message || '요약 데이터를 불러오지 못했습니다.');
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoaded(true));
  }, []);

  if (error) return <div className="page"><p>{error}</p></div>;
  if (!loaded) return <div className="page"><p>요약 데이터를 불러오는 중...</p></div>;
  if (!daily) {
    return (
      <div className="page">
        <h1>오늘의 요약</h1>
        <p className="subtitle">아직 측정 기록이 없어 체크리스트와 코멘트를 생성할 수 없어요.</p>
        <div className="card">
          <h3>측정 재가이드</h3>
          <p>조명/자세/얼굴 정렬을 맞춘 뒤 다시 측정해 주세요.</p>
          <button className="ghost" onClick={() => navigate('/measurement')}>측정하러 가기</button>
        </div>
      </div>
    );
  }

  const confidenceTrend = (daily.confidence_trend && daily.confidence_trend.length)
    ? daily.confidence_trend
    : [{ date: 'today', value: Number(daily.current_values?.confidence || 0) }];

  const toTimeLabel = (label, timestamp) => {
    const rawTs = String(timestamp || '');
    if (rawTs) {
      const hasExplicitTz = /([zZ]|[+\-]\d{2}:?\d{2})$/.test(rawTs);
      const normalizedTs = hasExplicitTz
        ? rawTs
        : `${rawTs.replace(' ', 'T').split('.')[0]}Z`;
      const d = new Date(normalizedTs);
      if (!Number.isNaN(d.getTime())) {
        const parts = new Intl.DateTimeFormat('ko-KR', {
          timeZone: 'Asia/Seoul',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        }).formatToParts(d);
        const map = Object.fromEntries(parts.map((p) => [p.type, p.value]));
        if (map.month && map.day && map.hour && map.minute) {
          return `${map.month}-${map.day} ${map.hour}:${map.minute}`;
        }
      }
    }

    const rawLabel = String(label || '');
    if (rawLabel.includes(':') && rawLabel.includes(' ')) return rawLabel;
    return rawLabel;
  };

  const combinedTrendData = (trends.combined_trend && trends.combined_trend.length)
    ? trends.combined_trend.map((d) => ({
      label: toTimeLabel(d.label, d.timestamp),
      timestamp: d.timestamp,
      bp: Number(d.bp || 0),
      glucose: Number(d.glucose || 0),
    }))
    : (trends.bp_trend || []).map((bpItem, idx) => ({
      label: toTimeLabel(bpItem.date, bpItem.timestamp),
      timestamp: bpItem.timestamp,
      bp: Number(bpItem.value || 0),
      glucose: Number((trends.glucose_trend || [])[idx]?.value || 0),
    }));

  const latestTimeLabel = combinedTrendData.length
    ? (combinedTrendData[combinedTrendData.length - 1].label || toTimeLabel('', combinedTrendData[combinedTrendData.length - 1].timestamp) || '-')
    : '-';

  const qualityChecklist = (daily.quality_checklist && daily.quality_checklist.length)
    ? daily.quality_checklist
    : [
      {
        id: 'lighting',
        label: '조명',
        status: Number(daily.current_values?.confidence || 0) >= 0.75 ? '좋음' : '보통',
        tip: '얼굴 정면에 균일한 빛이 들어오도록 조명을 조정해보세요.',
      },
      {
        id: 'movement',
        label: '움직임',
        status: '보통',
        tip: '측정 중에는 시선과 고개를 가능한 고정해 주세요.',
      },
      {
        id: 'alignment',
        label: '얼굴정렬',
        status: Number(daily.current_values?.confidence || 0) >= 0.75 ? '좋음' : '개선 필요',
        tip: '얼굴을 화면 중앙에 두고 턱선이 프레임 안에 들어오게 맞춰주세요.',
      },
    ];
  const qualitySourceText = daily.quality_meta?.is_measured_directly
    ? '실측 기반 (카메라 영상 지표)'
    : (daily.quality_checklist && daily.quality_checklist.length)
      ? '규칙 기반 추정 (신뢰도/변동성 기반)'
      : 'fallback 안내 (최신 측정에 실측 품질 데이터 없음)';

  return (
    <div className="page">
      <h1>오늘의 요약</h1>
      <p className="subtitle">최근 기록 흐름을 바탕으로 현재 상태를 참고용으로 정리했어요.</p>
      <div className="status-badge level">{daily.status.status_label}</div>
      <p>{daily.summary_text}</p>

      <div className="grid">
        <VitalCard
          type="blood-pressure"
          systolic={daily.current_values.bp_systolic}
          diastolic={daily.current_values.bp_diastolic}
          status={daily.status.status_label}
          trendData={(trends.bp_trend || []).map((d) => ({ value: Number(d.value || 0), date: d.date }))}
        />

        <VitalCard
          type="blood-sugar"
          value={daily.current_values.blood_sugar}
          status={daily.status.status_label}
          trendData={(trends.glucose_trend || []).map((d) => ({ value: Number(d.value || 0), date: d.date }))}
        />

        <div className="card confidence-card">
          <h3>측정 신뢰도 추이</h3>
          <div className="legend-row" style={{ marginBottom: 6 }}>
            <span className="legend-item"><span className="legend-swatch confidence" />신뢰도 라인</span>
            <span className="legend-item"><span className="legend-swatch confidence-bg" />배경(품질 참고 영역)</span>
          </div>
          <div style={{ width: '100%', height: 80 }}>
            <ResponsiveContainer>
              <LineChart data={confidenceTrend}>
                <YAxis domain={[0, 1]} hide />
                <Tooltip formatter={(v) => `${Math.round(Number(v || 0) * 100)}%`} labelFormatter={(l) => `${l}`} />
                <Line type="monotone" dataKey="value" stroke="#5C7CFA" dot={{ r: 2 }} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <small>
            최근 평균 신뢰도 {Math.round((confidenceTrend.reduce((a, c) => a + (c.value || 0), 0) /
              Math.max(1, confidenceTrend.length)) * 100)}%
          </small>
        </div>

        <div className="card">
          <h3>일주일 혈압/혈당 통합 추이</h3>
          <div className="row between">
            <div className="row gap">
              <span className={(trends.trend_percentages?.bp || 0) >= 0 ? 'trend up' : 'trend down'}>
                혈압 {(trends.trend_percentages?.bp || 0) >= 0 ? '▲' : '▼'} {Math.abs(trends.trend_percentages?.bp || 0).toFixed(1)}%
              </span>
              <span className={(trends.trend_percentages?.glucose || 0) >= 0 ? 'trend up' : 'trend down'}>
                혈당 {(trends.trend_percentages?.glucose || 0) >= 0 ? '▲' : '▼'} {Math.abs(trends.trend_percentages?.glucose || 0).toFixed(1)}%
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
              <YAxis yAxisId="left" domain={[90, 180]} />
              <YAxis yAxisId="right" orientation="right" domain={[60, 180]} />
              <Tooltip
                labelFormatter={(_, payload) => {
                  const row = payload && payload[0] ? payload[0].payload : null;
                  return row?.timestamp || row?.label || '-';
                }}
              />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="bp" name="혈압(SBP)" stroke="#5C7CFA" strokeWidth={2} dot={{ r: 2 }} />
              <Line yAxisId="right" type="monotone" dataKey="glucose" name="혈당" stroke="#9775FA" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
          <small>최근 측정 시각: {latestTimeLabel}</small>
        </div>

        <div className="card">
          <h3>측정 품질 체크리스트</h3>
          <small style={{ display: 'block', marginBottom: 8, opacity: 0.8 }}>
            출처: {qualitySourceText}
          </small>
          {qualityChecklist.map((item) => (
            <div key={item.id} className="check-item">
              <div className="row between">
                <strong>{item.label}</strong>
                <span className={`quality ${item.status === '좋음' ? 'good' : item.status === '보통' ? 'mid' : 'bad'}`}>{item.status}</span>
              </div>
              {Number.isFinite(Number(item.score)) && (
                <small>점수: {Math.round(Number(item.score) * 100)}%{' '}</small>
              )}
              <small>{item.tip}</small>
            </div>
          ))}
        </div>
      </div>

      <button onClick={() => navigate('/lifestyle')}>오늘의 생활 관리 제안 보기 →</button>
    </div>
  );
}

export default SummaryPage;
