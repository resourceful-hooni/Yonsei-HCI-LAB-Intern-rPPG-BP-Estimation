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
import { useLang } from '../contexts/LangContext';

function LifestyleGuidePage() {
  const { t } = useLang();
  const [comparison, setComparison] = useState(null);
  const [daily, setDaily] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [progress, setProgress] = useState(null);
  const [beforeAfter, setBeforeAfter] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [pageError, setPageError] = useState('');
  const [selected, setSelected] = useState(null);
  const [animTick, setAnimTick] = useState({ sleep: 0, water: 0, salt: 0 });
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
        setPageError(t('lg_load_error'));
      }
    });
  }, [t]);

  const deltaBadges = daily?.delta_badges || {
    bp_systolic: 0,
    blood_sugar: 0,
    bp_avg_7d: Number(daily?.current_values?.bp_systolic || 0),
    glucose_avg_7d: Number(daily?.current_values?.blood_sugar || 0),
  };

  const dailyComment = daily?.daily_comment
    || (Number(daily?.current_values?.confidence || 0) >= 0.75
      ? t('sp_subtitle')
      : t('sp_guide_body'));

  const handleCheck = async (habitId, checked) => {
    // Trigger pop animation by bumping the tick counter
    if (checked) {
      setAnimTick((prev) => ({ ...prev, [habitId]: prev[habitId] + 1 }));
    }
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

  const routineRow = (habitId, label, emoji) => {
    const item = progress?.progress?.[habitId];
    const isChecked = Boolean(progress?.today_checks?.[habitId]);
    return (
      <div className="routine-row" key={habitId}>
        <div className="routine-row-top">
          <label className="routine-check-label" style={{ cursor: 'pointer' }}>
            <span
              key={animTick[habitId]}
              className={animTick[habitId] > 0 ? 'check-pop' : ''}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
            >
              <input
                type="checkbox"
                checked={isChecked}
                onChange={(e) => handleCheck(habitId, e.target.checked)}
                style={{ width: 20, height: 20, cursor: 'pointer', flexShrink: 0 }}
              />
              <span>{emoji} {label}</span>
            </span>
          </label>
          <span className="routine-day-badge">
            {item?.completed_days ?? 0}<span style={{ opacity: 0.5 }}>/{item?.target_days ?? 7} {t('lg_day_suffix')}</span>
          </span>
        </div>
        <div className="progress-wrap" style={{ marginTop: 6 }}>
          <div className="progress" style={{ width: `${item?.progress ?? 0}%` }} />
        </div>
        <small style={{ opacity: 0.55, fontSize: '0.75rem' }}>{t('lg_checkin_hint')}</small>
      </div>
    );
  };

  return (
    <div className="page lifestyle-page">
      <h1>{t('lg_title')}</h1>
      <p className="subtitle">{t('lg_subtitle')}</p>
      {pageError && <p className="subtitle" style={{ color: '#c92a2a' }}>{pageError}</p>}

      {/* ── 섹션 1: 백분위 비교 ── */}
      {comparison ? (
        <div className="grid two ls-section">
          <ComparisonChart title={t('lg_bp_label')} percentile={comparison.bp_percentile} color="#5C7CFA" />
          <ComparisonChart title={t('lg_glucose_label')} percentile={comparison.glucose_percentile} color="#9775FA" />
        </div>
      ) : (
        <div className="ls-section"><div className="card"><p style={{ margin: 0 }}>{t('lg_bp_loading')}</p></div></div>
      )}

      {/* ── 섹션 2: 오늘 vs 7일 평균 + 한 줄 코멘트 ── */}
      {daily && (
        <div className="grid two ls-section">
          <div className="card">
            <h3>{t('lg_today_vs_7d')}</h3>
            <div className="row between" style={{ marginTop: 8 }}>
              <span>{t('lg_bp_sbp')}</span>
              <span className={`delta-badge ${deltaClass(deltaBadges.bp_systolic || 0)}`}>{deltaText(deltaBadges.bp_systolic || 0)}</span>
            </div>
            <small style={{ opacity: 0.55 }}>{t('lg_7d_avg', { v: Number(deltaBadges.bp_avg_7d || 0).toFixed(1) })}</small>
            <div className="row between" style={{ marginTop: 10 }}>
              <span>{t('lg_glucose')}</span>
              <span className={`delta-badge ${deltaClass(deltaBadges.blood_sugar || 0)}`}>{deltaText(deltaBadges.blood_sugar || 0)}</span>
            </div>
            <small style={{ opacity: 0.55 }}>{t('lg_7d_avg', { v: Number(deltaBadges.glucose_avg_7d || 0).toFixed(1) })}</small>
          </div>
          <div className="card">
            <h3>{t('lg_comment_title')}</h3>
            <p style={{ margin: 0 }}>{dailyComment}</p>
          </div>
        </div>
      )}

      {/* ── 섹션 3: 루틴 체크인 ── */}
      <div className="ls-section">
        <div className="card">
          <h3>{t('lg_routine_title')}</h3>
          <p className="subtitle" style={{ marginBottom: 14 }}>{t('lg_routine_rate')} <strong>{progress?.today_completion_rate ?? 0}%</strong> &nbsp;•&nbsp; {t('lg_routine_hint2')}</p>
          {routineRow('sleep', t('lg_sleep'), '💤')}
          {routineRow('water', t('lg_water'), '💧')}
          {routineRow('salt', t('lg_salt'), '🧂')}
        </div>
      </div>

      {/* ── 섹션 4: 주간 비교 + 알림 ── */}
      <div className="grid two ls-section">
        <div className="card">
          <h3>{t('lg_weekly_title')}</h3>
          {beforeAfter ? (
            <>
              <div className="row between" style={{ marginTop: 8 }}>
                <span>{t('lg_bp_sbp')}</span>
                <span>{beforeAfter.before?.bp_systolic} → <strong>{beforeAfter.after?.bp_systolic}</strong> ({beforeAfter.delta?.bp_systolic > 0 ? '+' : ''}{beforeAfter.delta?.bp_systolic})</span>
              </div>
              <div className="row between" style={{ marginTop: 8 }}>
                <span>{t('lg_glucose')}</span>
                <span>{beforeAfter.before?.blood_sugar} → <strong>{beforeAfter.after?.blood_sugar}</strong> ({beforeAfter.delta?.blood_sugar > 0 ? '+' : ''}{beforeAfter.delta?.blood_sugar})</span>
              </div>
            </>
          ) : <p style={{ margin: '8px 0 0', opacity: 0.6, fontSize: '0.9rem' }}>{t('lg_weekly_no_data')}</p>}
        </div>

        <div className="card">
          <h3>{t('lg_notif_title')}</h3>
          <small style={{ display: 'block', marginBottom: 10, opacity: 0.65, lineHeight: 1.45 }}>
            ℹ️ {t('lg_notif_sub')}
          </small>
          {notifications.length === 0 && (
            <p style={{ fontSize: '0.85rem', opacity: 0.55, margin: 0 }}>{t('lg_notif_empty')}</p>
          )}
          {notifications.map((item) => (
            <div key={item.setting_type} className="row between" style={{ paddingBottom: 8, borderBottom: '1px solid #f1f5f9' }}>
              <span style={{ fontSize: '0.9rem' }}>{item.setting_type === 'measurement' ? `⏰ ${t('lg_notif_measure')}` : item.setting_type === 'water' ? `💧 ${t('lg_notif_water')}` : `🌙 ${t('lg_notif_sleep')}`} <small style={{ opacity: 0.55 }}>({item.setting_time})</small></span>
              <input type="checkbox" checked={Boolean(item.enabled)} onChange={(e) => handleNotificationToggle(item, e.target.checked)} />
            </div>
          ))}
        </div>
      </div>

      {/* ── 섹션 5: 습관 제안 ── */}
      <div className="ls-section">
        <h3 className="section-title">{t('lg_habits_title')}</h3>
        <div className="grid">
          {recommendations.map((item) => (
            <HabitCard key={item.id} item={item} onOpen={() => setSelected(item)} />
          ))}
        </div>
      </div>

      <p className="disclaimer">ℹ️ {t('lg_disclaimer')}</p>

      {selected && <WhyNeededModal item={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

export default LifestyleGuidePage;
