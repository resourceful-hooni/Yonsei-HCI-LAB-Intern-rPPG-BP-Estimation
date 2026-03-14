function ComparisonChart({ title, percentile, color }) {
  return (
    <div className="card comparison-card">
      <h3>{title} 비교</h3>
      <div className="bar-wrap">
        <div className="bar low">낮음</div>
        <div className="bar normal">일반적인 범위</div>
        <div className="bar high">높음</div>
        <div className="marker" style={{ left: `${percentile}%`, borderColor: color }} />
      </div>
      <p className="percentile">내 위치: <strong>{percentile}%</strong></p>
    </div>
  );
}

export default ComparisonChart;
