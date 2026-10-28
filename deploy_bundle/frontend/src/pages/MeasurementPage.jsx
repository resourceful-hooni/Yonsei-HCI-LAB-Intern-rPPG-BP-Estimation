import { useState } from 'react';
import CameraView from '../components/Measurement/CameraView';
import ResultDisplay from '../components/Measurement/ResultDisplay';
import { useLang } from '../contexts/LangContext';

function MeasurementPage() {
  const [result, setResult] = useState(null);
  const { t } = useLang();

  return (
    <div className="page">
      <h1>{t('mp_title')}</h1>
      <p className="subtitle">{t('mp_subtitle')}</p>
      {!result ? (
        <CameraView userId="demo-user" onResult={setResult} />
      ) : (
        <ResultDisplay result={result} onRetry={() => setResult(null)} />
      )}
    </div>
  );
}

export default MeasurementPage;
