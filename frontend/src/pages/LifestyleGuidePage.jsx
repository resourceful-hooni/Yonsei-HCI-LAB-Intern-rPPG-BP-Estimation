import { useEffect, useState } from 'react';
import ComparisonChart from '../components/LifestyleGuide/ComparisonChart';
import HabitCard from '../components/LifestyleGuide/HabitCard';
import WhyNeededModal from '../components/LifestyleGuide/WhyNeededModal';
import {
  fetchBeforeAfter,
  fetchComparison,
  fetchDailySummary,
  fetchHabitProgress,
  fetchNotificationSettings,
  fetchRecommendations,
  saveHabitCheckin,
  saveNotificationSetting,
} from '../services/apiService';

function LifestyleGuidePage() {
  const [comparison, setComparison] = useState(null);
  const [daily, setDaily] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [progress, setProgress] = useState(null);
  const [beforeAfter, setBeforeAfter] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [pageError, setPageError] = useState('');
  const [selected, setSelected] = useState(null);
  const todayKey = new Date().toISOString().slice(0, 10);
  const offlineHabitKey = `visi_vital_habit_offline_${todayKey}`;

  const ensureProgressShape = (data) => ({
    today_completion_rate: data?.today_completion_rate ?? 0,
    today_checks: {
      sleep: Boolean(data?.today_checks?.sleep),
      water: Boolean(data?.today_checks?.water),
      salt: Boolean(data?.today_checks?.salt),
    },
    progress: {
      sleep: data?.progress?.sleep || { completed_days: 0, target_days: 7, progress: 0 },
      water: data?.progress?.water || { completed_days: 0, target_days: 7, progress: 0 },
      salt: data?.progress?.salt || { completed_days: 0, target_days: 7, progress: 0 },
    },
  });

  const deltaClass = (val) => (val > 0 ? 'delta-up' : val < 0 ? 'delta-down' : 'delta-flat');
  const deltaText = (val) => `${val > 0 ? '+' : ''}${Number(val || 0).toFixed(1)}`;

  useEffect(() => {
    Promise.allSettled([
      fetchDailySummary('demo-user'),
      fetchComparison('demo-user'),
      fetchRecommendations('demo-user'),
      fetchHabitProgress(7),
      fetchBeforeAfter(14),
      fetchNotificationSettings(),
    ]).then(([d, c, r, p, b, n]) => {
      if (d.status === 'fulfilled') setDaily(d.value.data || null);
      if (c.status === 'fulfilled') setComparison(c.value.data);
      if (r.status === 'fulfilled') setRecommendations(r.value.data || []);
      if (p.status === 'fulfilled') {
        const serverProgress = ensureProgressShape(p.value.data || null);
        let offlineDraft = {};
        try {
          offlineDraft = JSON.parse(localStorage.getItem(offlineHabitKey) || '{}');
        } catch {
          offlineDraft = {};
        }
        const merged = {
          ...serverProgress,
          today_checks: {
            ...serverProgress.today_checks,
            sleep: Boolean(offlineDraft.sleep ?? serverProgress.today_checks.sleep),
            water: Boolean(offlineDraft.water ?? serverProgress.today_checks.water),
            salt: Boolean(offlineDraft.salt ?? serverProgress.today_checks.salt),
          }
        };
        const doneCount = Object.values(merged.today_checks).filter(Boolean).length;
        merged.today_completion_rate = Number(((doneCount / 3) * 100).toFixed(1));
        setProgress(merged);
      }
      if (b.status === 'fulfilled') setBeforeAfter(b.value.data || null);
      if (n.status === 'fulfilled') setNotifications(n.value.data || []);

      if (c.status === 'rejected' && r.status === 'rejected') {
        setPageError('생활관리 데이터를 불러오지 못했습니다. 백엔드 서버 상태를 확인해주세요.');
      }
    });
  }, []);

  const deltaBadges = daily?.delta_badges || {
    bp_systolic: 0,
    blood_sugar: 0,
    bp_avg_7d: Number(daily?.current_values?.bp_systolic || 0),
    glucose_avg_7d: Number(daily?.current_values?.blood_sugar || 0),
  };

  const dailyComment = daily?.daily_comment
    || (Number(daily?.current_values?.confidence || 0) >= 0.75
      ? '오늘 측정 품질이 비교적 안정적이에요. 같은 환경에서 꾸준히 측정해보세요.'
      : '측정 품질을 높이기 위해 조명과 자세를 일정하게 맞춘 뒤 다시 측정해보세요.');

  const handleCheck = async (habitId, checked) => {
    const prev = progress;
    const draft = ensureProgressShape(prev);
    const wasChecked = Boolean(draft.today_checks[habitId]);
    draft.today_checks[habitId] = checked;
    const doneCount = Object.values(draft.today_checks).filter(Boolean).length;
    draft.today_completion_rate = Number(((doneCount / 3) * 100).toFixed(1));

    const item = draft.progress[habitId] || { completed_days: 0, target_days: 7, progress: 0 };
    const nextCompleted = Math.max(0, Math.min(item.target_days, item.completed_days + (checked && !wasChecked ? 1 : (!checked && wasChecked ? -1 : 0))));
    draft.progress[habitId] = {
      ...item,
      completed_days: nextCompleted,
      progress: Number(((nextCompleted / Math.max(1, item.target_days)) * 100).toFixed(1)),
    };
    setProgress(draft);

    try {
      await saveHabitCheckin(habitId, checked);
      try {
        const curr = JSON.parse(localStorage.getItem(offlineHabitKey) || '{}');
        curr[habitId] = checked;
        localStorage.setItem(offlineHabitKey, JSON.stringify(curr));
      } catch {
        // no-op
      }
      const p = await fetchHabitProgress(7);
      setProgress(ensureProgressShape(p.data || null));
    } catch (err) {
      try {
        const curr = JSON.parse(localStorage.getItem(offlineHabitKey) || '{}');
        curr[habitId] = checked;
        localStorage.setItem(offlineHabitKey, JSON.stringify(curr));
      } catch {
        // no-op
      }
      setProgress(draft);
    }
  };

  const handleNotificationToggle = async (item, enabled) => {
    try {
      await saveNotificationSetting({ ...item, enabled });
      const n = await fetchNotificationSettings();
      setNotifications(n.data || []);
    } catch (err) {
      setPageError(err.message);
    }
  };

  const progressItem = (id, title) => {
    const item = progress?.progress?.[id];
    return (
      <div className="routine-item" key={id}>
        <div className="row between">
          <strong>{title}</strong>
          <small>{item?.completed_days ?? 0}/{item?.target_days ?? 7}일</small>
        </div>
        <div className="progress-wrap" style={{ marginTop: 6 }}>
          <div className="progress" style={{ width: `${item?.progress ?? 0}%` }} />
        </div>
      </div>
    );
  };

  const checkToggle = (habitId, label) => (
    <label className="check-toggle" key={habitId}>
      <input
        type="checkbox"
        checked={Boolean(progress?.today_checks?.[habitId])}
        onChange={(e) => handleCheck(habitId, e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );

  return (
    <div className="page">
      <h1>생활관리 참고 가이드</h1>
      <p className="subtitle">최근 기록을 일반적인 범위와 비교해 보여드려요.</p>
      {pageError && <p className="subtitle" style={{ color: '#c92a2a' }}>{pageError}</p>}

      {comparison && (
        <div className="grid two">
          <ComparisonChart title="혈압" percentile={comparison.bp_percentile} color="#5C7CFA" />
          <ComparisonChart title="혈당" percentile={comparison.glucose_percentile} color="#9775FA" />
        </div>
      )}

      {!comparison && <div className="card"><p>혈압/혈당 비교 데이터를 불러오는 중이거나 서버 연결이 필요합니다.</p></div>}

      {daily && (
        <div className="grid two">
          <div className="card">
            <h3>오늘 vs 최근 7일 평균</h3>
            <div className="row between">
              <span>혈압(SBP)</span>
              <span className={`delta-badge ${deltaClass(deltaBadges.bp_systolic || 0)}`}>
                {deltaText(deltaBadges.bp_systolic || 0)}
              </span>
            </div>
            <small>7일 평균 {Number(deltaBadges.bp_avg_7d || 0).toFixed(1)} 기준</small>
            <div className="row between" style={{ marginTop: 8 }}>
              <span>혈당</span>
              <span className={`delta-badge ${deltaClass(deltaBadges.blood_sugar || 0)}`}>
                {deltaText(deltaBadges.blood_sugar || 0)}
              </span>
            </div>
            <small>7일 평균 {Number(deltaBadges.glucose_avg_7d || 0).toFixed(1)} 기준</small>
          </div>

          <div className="card">
            <h3>오늘 한 줄 코멘트</h3>
            <p>{dailyComment}</p>
          </div>
        </div>
      )}

      <div className="grid two">
        <div className="card">
          <h3>추천 실천률 체크</h3>
          <p className="subtitle" style={{ marginBottom: 8 }}>오늘 실천률: {progress?.today_completion_rate ?? 0}%</p>
          {checkToggle('sleep', '수면 루틴 실천')}
          {checkToggle('water', '수분 루틴 실천')}
          {checkToggle('salt', '저염 루틴 실천')}
        </div>

        <div className="card">
          <h3>개인 루틴 목표 3개</h3>
          {progressItem('sleep', '수면')}
          {progressItem('water', '수분')}
          {progressItem('salt', '염분')}
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>주간 변화 전/후 비교</h3>
          {beforeAfter ? (
            <>
              <p>혈압(SBP): {beforeAfter.before?.bp_systolic} → {beforeAfter.after?.bp_systolic} ({beforeAfter.delta?.bp_systolic > 0 ? '+' : ''}{beforeAfter.delta?.bp_systolic})</p>
              <p>혈당: {beforeAfter.before?.blood_sugar} → {beforeAfter.after?.blood_sugar} ({beforeAfter.delta?.blood_sugar > 0 ? '+' : ''}{beforeAfter.delta?.blood_sugar})</p>
            </>
          ) : <p>비교를 위한 데이터가 아직 부족해요.</p>}
        </div>

        <div className="card">
          <h3>알림 설정</h3>
          {notifications.map((item) => (
            <div key={item.setting_type} className="row between" style={{ marginBottom: 8 }}>
              <span>{item.setting_type === 'measurement' ? '측정 시간' : item.setting_type === 'water' ? '물 마시기' : '취침 루틴'} ({item.setting_time})</span>
              <input
                type="checkbox"
                checked={Boolean(item.enabled)}
                onChange={(e) => handleNotificationToggle(item, e.target.checked)}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="grid">
        <h3 className="section-title">습관 제안</h3>
        {recommendations.map((item) => (
          <HabitCard key={item.id} item={item} onOpen={() => setSelected(item)} />
        ))}
      </div>

      <p className="disclaimer">ℹ️ 본 서비스는 의료인의 진단이나 치료를 대체하지 않으며, 제공되는 정보는 자가 건강관리를 위한 참고 정보입니다.</p>

      {selected && <WhyNeededModal item={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

export default LifestyleGuidePage;
