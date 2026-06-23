const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '/api';
const API_KEY = process.env.REACT_APP_API_KEY || 'your-frontend-api-key';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Localized error messages (this module has no access to the React i18n context,
// so it reads the persisted language directly — keeps EN users from seeing KR).
const curLang = () => {
  try { return localStorage.getItem('visi_lang') === 'en' ? 'en' : 'ko'; } catch (_) { return 'ko'; }
};
const ERR = {
  conn: {
    ko: '서버에 연결하지 못했습니다. 백엔드 서버(5000)가 실행 중인지 확인해주세요.',
    en: 'Could not reach the server. Make sure the backend (port 5000) is running.',
  },
  tooLarge: {
    ko: '전송 데이터가 너무 큽니다. 자동 압축을 적용했지만 네트워크 상태에 따라 실패할 수 있어요. 다시 시도해주세요.',
    en: 'The upload is too large. Auto-compression was applied, but it may still fail depending on your network — please try again.',
  },
  failed: { ko: 'API 요청에 실패했습니다.', en: 'The API request failed.' },
};
const errMsg = (k) => ERR[k][curLang()];

const request = async (url, options = {}) => {
  // Retry transient network failures for safe (GET) requests only — never retry
  // POSTs automatically (they create/mutate state). Pass { retries } to override.
  const method = (options.method || 'GET').toUpperCase();
  const retries = options.retries != null ? options.retries : (method === 'GET' ? 2 : 0);

  let response;
  let lastNetErr = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      response = await fetch(url, {
        ...options,
        cache: 'no-store',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
          ...(options.headers || {})
        }
      });
      lastNetErr = null;
      break;
    } catch (err) {
      lastNetErr = err;
      if (attempt < retries) {
        await sleep(400 * (attempt + 1));
      }
    }
  }
  if (lastNetErr) {
    throw new Error(errMsg('conn'));
  }

  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : {};
  if (!response.ok) {
    if (response.status === 413) {
      throw new Error(errMsg('tooLarge'));
    }
    throw new Error(data.error || errMsg('failed'));
  }
  return data;
};

export const startMeasurement = async (userId) => {
  return request(`${API_BASE_URL}/measurement/start`, {
    method: 'POST',
    body: JSON.stringify({ user_id: userId })
  });
};

export const processMeasurement = async (measurementId, frames, frameRate, qualityMetrics, frameTimestamps) => {
  return request(`${API_BASE_URL}/measurement/process`, {
    method: 'POST',
    body: JSON.stringify({
      measurement_id: measurementId,
      frames,
      frame_rate: frameRate,
      // Real per-frame capture times (seconds) so the backend can use the true
      // sampling rate instead of the nominal frameRate (setInterval is jittery).
      frame_timestamps: Array.isArray(frameTimestamps) ? frameTimestamps : null,
      quality_metrics: qualityMetrics || null
    })
  });
};

export const fetchDailySummary = async (userId) => {
  return request(`${API_BASE_URL}/summary/daily?user_id=${encodeURIComponent(userId)}`);
};

export const fetchTrends = async (userId, days = 7) => {
  return request(`${API_BASE_URL}/summary/trends?user_id=${encodeURIComponent(userId)}&days=${days}&_t=${Date.now()}`);
};

export const fetchComparison = async (userId) => {
  return request(`${API_BASE_URL}/lifestyle/comparison?user_id=${encodeURIComponent(userId)}`);
};

export const fetchRecommendations = async (userId) => {
  return request(`${API_BASE_URL}/lifestyle/recommendations?user_id=${encodeURIComponent(userId)}`);
};

export const saveHabitCheckin = async (habitId, completed, date) => {
  return request(`${API_BASE_URL}/lifestyle/habits/checkin`, {
    method: 'POST',
    body: JSON.stringify({ habit_id: habitId, completed, date })
  });
};

export const fetchHabitProgress = async (days = 7) => {
  return request(`${API_BASE_URL}/lifestyle/habits/progress?days=${days}&_t=${Date.now()}`);
};

export const fetchBeforeAfter = async (days = 14) => {
  return request(`${API_BASE_URL}/lifestyle/before-after?days=${days}`);
};

export const fetchNotificationSettings = async () => {
  return request(`${API_BASE_URL}/lifestyle/notifications`);
};

export const saveNotificationSetting = async (payload) => {
  return request(`${API_BASE_URL}/lifestyle/notifications`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
};
