import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

function TrendChart({ title, data, color, min, max, trend }) {
  const gradientId = `gradient-${title.replace(/\s+/g, '-')}`;

  return (
    <div className="card chart">
      <div className="row between">
        <h3>{title}</h3>
        <span className={trend >= 0 ? 'trend up' : 'trend down'}>
          {trend >= 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(1)}%
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.45} />
              <stop offset="95%" stopColor={color} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={[min, max]} />
          <Tooltip />
          <Area type="monotone" dataKey="value" stroke={color} fill={`url(#${gradientId})`} />
        </AreaChart>
      </ResponsiveContainer>
      <div className="row gap">
        <span>⊙ 오늘</span>
        <span>○ Weekly</span>
      </div>
    </div>
  );
}

export default TrendChart;
