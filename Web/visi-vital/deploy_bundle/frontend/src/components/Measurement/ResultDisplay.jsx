import { useNavigate } from 'react-router-dom';

function ResultDisplay({ result, onRetry }) {
  const navigate = useNavigate();

  const confidencePercent = Math.round((result.confidence || 0) * 100);

  const getGlucoseStatus = (value) => {
    if (value < 90) {
      return {
        label: '낮은 편',
        description: '최근 값이 일반 참고 범위보다 낮은 편으로 보여요. 식사/수면/활동 패턴을 함께 확인해보세요.'
      };
    }
    if (value <= 110) {
      return {
        label: '일반적인 범위',
        description: '최근 값이 일반적인 참고 범위에 있어요. 현재 루틴을 유지하며 추이를 함께 보세요.'
      };
    }
    return {
      label: '높은 편',
      description: '최근 값이 일반 참고 범위보다 높은 편으로 보여요. 식사/수분/활동 패턴을 점검해보세요.'
    };
  };

  const glucoseStatus = getGlucoseStatus(Number(result.blood_sugar || 0));

  return (
    <div className="card">
      <h2>측정 결과</h2>
      <div className="result-number">{result.bp_systolic} / {result.bp_diastolic}</div>
      <div className="result-number">{result.blood_sugar}</div>

      <p>신뢰도: {confidencePercent}%</p>
      <p className="subtitle">신뢰도 기준: 얼굴 ROI 유효 프레임 비율 + rPPG 신호 품질 점수 기반</p>

      <p className="subtitle">
        혈압 추정 소스: 내 혈압 AI 모델(MS-TCN + Linear Attention, rPPG 특화)
        {result.bp_source === 'research_model' ? ' 사용 중' : ' 폴백 모드'}
      </p>

      <p className="subtitle">혈당 값 소스: 더미데이터(참고용)</p>

      <div className="card" style={{ marginTop: 12, background: '#fafbff' }}>
        <h3 style={{ marginTop: 0 }}>혈당 상태 설명</h3>
        <p><strong>{glucoseStatus.label}</strong></p>
        <p style={{ marginBottom: 0 }}>{glucoseStatus.description}</p>
      </div>

      <div className="actions">
        <button onClick={() => navigate('/summary')}>오늘의 요약 보기</button>
        <button className="ghost" onClick={onRetry}>다시 측정</button>
      </div>
    </div>
  );
}

export default ResultDisplay;
