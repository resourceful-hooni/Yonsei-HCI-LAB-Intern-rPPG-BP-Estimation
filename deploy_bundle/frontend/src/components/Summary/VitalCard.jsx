import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts';

function VitalCard({ type, systolic, diastolic, value, status, trendData = [] }) {
  const isBP = type === 'blood-pressure';
  const title = isBP ? '혈압' : '혈당';
  const icon = isBP ? '💧' : '🩸';
  const display = isBP ? `${systolic} / ${diastolic}` : `${value}`;
  const tone = status === '안정적' ? 'stable' : status === '관심 필요' ? 'watch' : 'care';
  const values = (trendData || []).map((d) => Number(d.value || 0));
  const localMin = values.length ? Math.min(...values) : 0;
  const localMax = values.length ? Math.max(...values) : 1;
  const yMin = Math.max(0, localMin - Math.max(2, (localMax - localMin) * 0.2));
  const yMax = localMax + Math.max(2, (localMax - localMin) * 0.2);

  return (
    <div className={`card vital-card ${tone}`}>
      <div className="row between">
        <h3>{icon} {title}</h3>
        <span className="status-badge">{status}</span>
      </div>
      <div className="legend-row" style={{ marginBottom: 6 }}>
        <span className="legend-item">
          <span className={`legend-swatch ${isBP ? 'bp' : 'glucose'}`} />추이 라인
        </span>
        <span className={`legend-pill ${tone}`}>배경: {status}</span>
      </div>
      <div className="result-number">{display}</div>
      <div style={{ width: '100%', height: 64, marginTop: 4 }}>
        <ResponsiveContainer>
          <LineChart data={trendData}>
            <YAxis hide domain={[yMin, yMax]} />
            <Tooltip formatter={(v) => Number(v).toFixed(1)} labelFormatter={(l) => `${l}`} />
            <Line type="monotone" dataKey="value" stroke={isBP ? '#5C7CFA' : '#9775FA'} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <button className="ghost">ℹ️ 설명</button>
    </div>
  );
}

export default VitalCard;
