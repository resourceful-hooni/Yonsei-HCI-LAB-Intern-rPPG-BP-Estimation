import { useState } from 'react';
import CameraView from '../components/Measurement/CameraView';
import ResultDisplay from '../components/Measurement/ResultDisplay';

function MeasurementPage() {
  const [result, setResult] = useState(null);

  return (
    <div className="page">
      <h1>10초 측정</h1>
      <p className="subtitle">조용한 환경에서 얼굴을 중앙에 두고 편안하게 측정해보세요.</p>
      {!result ? (
        <CameraView userId="demo-user" onResult={setResult} />
      ) : (
        <ResultDisplay result={result} onRetry={() => setResult(null)} />
      )}
    </div>
  );
}

export default MeasurementPage;
