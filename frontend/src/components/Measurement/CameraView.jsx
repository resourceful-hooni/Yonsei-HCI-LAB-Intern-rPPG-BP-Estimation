import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FaceMesh } from '@mediapipe/face_mesh';
import { processMeasurement, startMeasurement } from '../../services/apiService';
import { useLang } from '../../contexts/LangContext';

const TOTAL_FRAMES = 300;
const FPS = 30;
const CAPTURE_WIDTH = 320;
const JPEG_QUALITY = 0.55;
const MEDIAPIPE_WEBGL_LOG_PATTERN = /(gl_context_webgl\.cc:151|gl_context\.cc:351|gl_context\.cc:821|Successfully created a WebGL context|GL version:|OpenGL error checking is disabled)/;

function suppressMediapipeWebglLogs() {
  const original = {
    log: console.log,
    info: console.info,
    warn: console.warn,
  };

  const shouldSuppress = (args) => {
    const message = args
      .map((entry) => {
        if (typeof entry === 'string') return entry;
        if (entry instanceof Error) return entry.message;
        return String(entry ?? '');
      })
      .join(' ');
    return MEDIAPIPE_WEBGL_LOG_PATTERN.test(message);
  };

  console.log = (...args) => {
    if (shouldSuppress(args)) return;
    original.log(...args);
  };
  console.info = (...args) => {
    if (shouldSuppress(args)) return;
    original.info(...args);
  };
  console.warn = (...args) => {
    if (shouldSuppress(args)) return;
    original.warn(...args);
  };

  return () => {
    console.log = original.log;
    console.info = original.info;
    console.warn = original.warn;
  };
}

function CameraView({ userId, onResult }) {
  const { t } = useLang();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const timerRef = useRef(null);
  const rafRef = useRef(null);
  const faceMeshRef = useRef(null);
  const initializedRef = useRef(false);
  const processingRef = useRef(false);
  const isMeasuringRef = useRef(false);
  const qualityRef = useRef({
    lightingSamples: [],
    movementSamples: [],
    alignmentSamples: [],
    detectedFrames: 0,
    sampledFrames: 0,
  });
  const prevFaceCenterRef = useRef(null);
  const latestFaceMetricsRef = useRef({ hasFace: false, movementScore: null, alignmentScore: null });

  const [permissionError, setPermissionError] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceMeshEnabled, setFaceMeshEnabled] = useState(true);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('cam_msg_no_face');
  const [measurementFailed, setMeasurementFailed] = useState(false);
  const [failReasons, setFailReasons] = useState([]);
  const [liveQuality, setLiveQuality] = useState({
    lighting: { score: null, status: 'wait', tip: t('cam_tip_lighting_wait') },
    movement: { score: null, status: 'wait', tip: t('cam_tip_movement_wait') },
    alignment: { score: null, status: 'wait', tip: t('cam_tip_alignment_wait') },
  });

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));

  const statusFromScore = (score) => {
    if (score === null || Number.isNaN(Number(score))) return 'wait';
    const v = Number(score);
    if (v >= 0.75) return 'good';
    if (v >= 0.55) return 'mid';
    return 'bad';
  };

  const estimateLightingScore = (ctx, w, h) => {
    try {
      const step = 16;
      const image = ctx.getImageData(0, 0, w, h).data;
      let sum = 0;
      let count = 0;
      for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
          const idx = (y * w + x) * 4;
          const r = image[idx];
          const g = image[idx + 1];
          const b = image[idx + 2];
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          sum += luma;
          count += 1;
        }
      }
      const mean = count > 0 ? sum / count : 0;
      const target = 145;
      const deviation = Math.abs(mean - target);
      return clamp01(1 - deviation / 110);
    } catch {
      return 0.5;
    }
  };

  useEffect(() => {
    isMeasuringRef.current = isMeasuring;
  }, [isMeasuring]);

  useEffect(() => {
    if (initializedRef.current) return undefined;
    initializedRef.current = true;

    let unmounted = false;
    const restoreConsole = suppressMediapipeWebglLogs();

    const init = async () => {
      try {
        faceMeshRef.current = new FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`
        });
        faceMeshRef.current.setOptions({
          maxNumFaces: 1,
          refineLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
      } catch (err) {
        setFaceMeshEnabled(false);
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        // FaceMesh 실패 시 측정 기능은 계속 사용 가능하도록 fallback
        if (!faceMeshRef.current || !faceMeshEnabled) {
          setFaceDetected(true);
          setIsReady(true);
          setMessage('fl_ready');
          return;
        }

        faceMeshRef.current.onResults((results) => {
          const overlay = overlayRef.current;
          const video = videoRef.current;
          if (!overlay) return;
          if (!video) return;

          const ctx = overlay.getContext('2d');
          ctx.clearRect(0, 0, overlay.width, overlay.height);

          const hasFace = !!results.multiFaceLandmarks?.length;
          setFaceDetected(hasFace);

          if (!isMeasuringRef.current) {
            if (hasFace) {
              setMessage('fl_ready');
            } else {
              setMessage('cam_msg_no_face');
            }
          }

          if (hasFace) {
            const points = results.multiFaceLandmarks[0];

            let cx = 0;
            let cy = 0;
            points.forEach((pt) => {
              cx += pt.x;
              cy += pt.y;
            });
            cx /= points.length;
            cy /= points.length;

            const distCenter = Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2);
            const alignmentScore = clamp01(1 - distCenter / 0.38);

            let movementScore = 1;
            if (prevFaceCenterRef.current) {
              const dx = cx - prevFaceCenterRef.current.x;
              const dy = cy - prevFaceCenterRef.current.y;
              const delta = Math.sqrt(dx * dx + dy * dy);
              movementScore = clamp01(1 - delta / 0.03);
            }
            prevFaceCenterRef.current = { x: cx, y: cy };
            latestFaceMetricsRef.current = { hasFace: true, movementScore, alignmentScore };

            const liveCanvas = canvasRef.current;
            const liveCtx = liveCanvas?.getContext('2d', { willReadFrequently: true });
            let lightingScore = null;
            if (video && liveCanvas && liveCtx) {
              const srcW = video.videoWidth || 640;
              const srcH = video.videoHeight || 480;
              const w = 160;
              const h = Math.max(1, Math.round((srcH / Math.max(1, srcW)) * w));
              liveCanvas.width = w;
              liveCanvas.height = h;
              liveCtx.drawImage(video, 0, 0, w, h);
              lightingScore = estimateLightingScore(liveCtx, w, h);
            }

            setLiveQuality({
              lighting: {
                score: lightingScore,
                status: statusFromScore(lightingScore),
                tip: t('sp_q_lighting') + ': ' + t('cam_quality_warn'),
              },
              movement: {
                score: movementScore,
                status: statusFromScore(movementScore),
                tip: t('cam_fail_tip2'),
              },
              alignment: {
                score: alignmentScore,
                status: statusFromScore(alignmentScore),
                tip: t('cam_fail_tip1'),
              },
            });

            // object-fit: cover 기준으로 좌표 보정
            const videoW = video.videoWidth || 640;
            const videoH = video.videoHeight || 480;
            const scale = Math.max(overlay.width / videoW, overlay.height / videoH);
            const drawW = videoW * scale;
            const drawH = videoH * scale;
            const offsetX = (overlay.width - drawW) / 2;
            const offsetY = (overlay.height - drawH) / 2;

            // x좌표를 미러링: 비디오가 CSS scaleX(-1)이므로 랜드마크도 반전
            ctx.strokeStyle = '#5C7CFA';
            ctx.lineWidth = 1;
            points.forEach((pt) => {
              const rawX = pt.x * videoW * scale + offsetX;
              const x = overlay.width - rawX; // mirror x
              const y = pt.y * videoH * scale + offsetY;
              ctx.beginPath();
              ctx.arc(x, y, 1, 0, Math.PI * 2);
              ctx.stroke();
            });
          }
          if (!hasFace) {
            prevFaceCenterRef.current = null;
            latestFaceMetricsRef.current = { hasFace: false, movementScore: null, alignmentScore: null };
            setLiveQuality((prev) => ({
              lighting: prev.lighting,
              movement: {
                score: null,
                status: 'wait',
                tip: t('cam_msg_no_face'),
              },
              alignment: {
                score: null,
                status: 'wait',
                tip: t('cam_msg_no_face'),
              },
            }));
          }
        });

        const process = async () => {
          if (unmounted) return;

          const video = videoRef.current;
          const overlay = overlayRef.current;
          if (!video || !overlay || !faceMeshRef.current) return;

          if (processingRef.current) {
            rafRef.current = requestAnimationFrame(process);
            return;
          }

          // Use rendered element size for proper alignment with CSS-scaled video
          const drawW = Math.max(1, video.clientWidth || video.videoWidth || 640);
          const drawH = Math.max(1, video.clientHeight || video.videoHeight || 480);
          if (overlay.width !== drawW || overlay.height !== drawH) {
            overlay.width = drawW;
            overlay.height = drawH;
          }

          try {
            processingRef.current = true;
            await faceMeshRef.current.send({ image: video });
          } catch (err) {
            setFaceMeshEnabled(false);
            setFaceDetected(true);
            setMessage('cam_msg_guide_load');
            return;
          } finally {
            processingRef.current = false;
          }

          rafRef.current = requestAnimationFrame(process);
        };

        rafRef.current = requestAnimationFrame(process);
        setIsReady(true);
      }
    };

    init().catch((err) => {
      const name = err?.name || '';
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setPermissionError(true);
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setMessage('cam_msg_no_cam');
      } else {
        setMessage(err?.message || 'cam_initializing');
      }
    });

    return () => {
      unmounted = true;
      restoreConsole();
      if (timerRef.current) clearInterval(timerRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const stream = videoRef.current?.srcObject;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    const srcW = video.videoWidth || 640;
    const srcH = video.videoHeight || 480;
    const scale = CAPTURE_WIDTH / srcW;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.max(1, Math.round(srcH * scale));
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', JPEG_QUALITY);
  };

  const startCapture = async () => {
    if (!isReady || isMeasuring) return;
    if (faceMeshEnabled && !faceDetected) {
      setMessage('cam_msg_no_face');
      return;
    }

    setIsMeasuring(true);
    setMessage('fl_measuring_good');
    qualityRef.current = {
      lightingSamples: [],
      movementSamples: [],
      alignmentSamples: [],
      detectedFrames: 0,
      sampledFrames: 0,
    };
    prevFaceCenterRef.current = null;

    try {
      const startRes = await startMeasurement(userId);
      const measurementId = startRes.measurement_id;
      const frames = [];

      timerRef.current = setInterval(async () => {
        const frame = captureFrame();
        if (frame) {
          frames.push(frame);
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext('2d', { willReadFrequently: true });
          if (canvas && ctx) {
            const lightingScore = estimateLightingScore(ctx, canvas.width, canvas.height);
            qualityRef.current.lightingSamples.push(lightingScore);
          }

          qualityRef.current.sampledFrames += 1;
          if (latestFaceMetricsRef.current.hasFace) {
            qualityRef.current.detectedFrames += 1;
            if (latestFaceMetricsRef.current.movementScore !== null) {
              qualityRef.current.movementSamples.push(latestFaceMetricsRef.current.movementScore);
            }
            if (latestFaceMetricsRef.current.alignmentScore !== null) {
              qualityRef.current.alignmentSamples.push(latestFaceMetricsRef.current.alignmentScore);
            }
          }
        }

        const next = Math.min(frames.length, TOTAL_FRAMES);
        setProgress(Math.round((next / TOTAL_FRAMES) * 100));

        if (frames.length >= TOTAL_FRAMES) {
          clearInterval(timerRef.current);
          setMessage('cam_msg_analysing');

          const avg = (arr) => arr.length ? arr.reduce((a, c) => a + c, 0) / arr.length : null;
          const qualityMetrics = {
            lighting_score: avg(qualityRef.current.lightingSamples),
            movement_score: avg(qualityRef.current.movementSamples),
            alignment_score: avg(qualityRef.current.alignmentSamples),
            face_detected_ratio: qualityRef.current.sampledFrames
              ? qualityRef.current.detectedFrames / qualityRef.current.sampledFrames
              : null,
            method: 'frontend_vision',
          };

          // 품질 기준 미달 시 재측정 요구
          const reasons = [];
          const fdr = qualityMetrics.face_detected_ratio;
          if (fdr !== null && fdr < 0.4) {
            reasons.push(t('cam_fail_reason_face', { v: Math.round(fdr * 100) }));
          }
          if (qualityMetrics.movement_score !== null && qualityMetrics.movement_score < 0.35) {
            reasons.push(t('cam_fail_reason_move'));
          }
          if (qualityMetrics.lighting_score !== null && qualityMetrics.lighting_score < 0.35) {
            reasons.push(t('cam_fail_reason_light'));
          }
          if (reasons.length > 0) {
            setFailReasons(reasons);
            setMeasurementFailed(true);
            setIsMeasuring(false);
            setProgress(0);
            return;
          }

          const res = await processMeasurement(measurementId, frames, FPS, qualityMetrics);
          onResult(res.data);
          setMessage('cam_msg_after');
          setIsMeasuring(false);
        }
      }, Math.round(1000 / FPS));
    } catch (err) {
      if ((err.message || '').includes('Payload Too Large') || (err.message || '').includes('전송 데이터가 너무 큽니다')) {
        setMessage('cam_msg_too_large');
      } else {
        setMessage(err.message || 'cam_msg_error');
      }
      setIsMeasuring(false);
    }
  };

  // 품질 미달으로 인한 재측정 요구 화면
  if (measurementFailed) {
    return (
      <div className="card measurement-failed-card">
        <div className="failed-icon-svg" aria-hidden="true">
          <svg viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="28" cy="28" r="25" stroke="#f97316" strokeWidth="2.5" />
            <path d="M28 17v14" stroke="#f97316" strokeWidth="3" strokeLinecap="round" />
            <circle cx="28" cy="38" r="2.2" fill="#f97316" />
          </svg>
        </div>
        <h3>{t('cam_fail_title')}</h3>
        <p style={{ marginBottom: 16 }}>{t('cam_fail_body')}</p>
        <ul className="fail-reason-list">
          {failReasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
        <div className="fail-tips">
          <strong>{t('cam_fail_tips')}</strong>
          <ul>
            <li>{t('cam_fail_tip1')}</li>
            <li>{t('cam_fail_tip2')}</li>
            <li>{t('cam_fail_tip3')}</li>
          </ul>
        </div>
        <button onClick={() => { setMeasurementFailed(false); setFailReasons([]); setProgress(0); }}
          style={{ marginTop: 8 }}>
          {t('cam_fail_retry')}
        </button>
      </div>
    );
  }

  if (permissionError) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '32px 16px' }}>
        <div className="permission-icon-svg" aria-hidden="true">
          <svg viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="8" y="16" width="40" height="30" rx="5" stroke="#94a3b8" strokeWidth="2.2" />
            <circle cx="28" cy="31" r="8" stroke="#94a3b8" strokeWidth="2.2" />
            <path d="M22 16l6 4 6-4" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
            <line x1="38" y1="16" x2="18" y2="40" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
        </div>
        <h3 style={{ margin: '0 0 8px' }}>{t('cam_perm_title')}</h3>
        <p style={{ marginBottom: 16 }}>
          {t('cam_perm_body1')}<br />
          {t('cam_perm_body2')}<br />
          {t('cam_perm_body3')}
        </p>
        <button onClick={() => window.location.reload()}>{t('cam_perm_refresh')}</button>
      </div>
    );
  }

  const RING_R = 38;
  const RING_C = 2 * Math.PI * RING_R; // ≈ 238.76

  return (
    <>
    <div className="card">
      <div className="camera-wrap">
        <video ref={videoRef} className="camera" muted playsInline style={{ transform: 'scaleX(-1)' }} />
        <canvas ref={overlayRef} className="overlay" />
        {/* 상황별 얼굴 감지 안내 메시지 */}
        {isReady && (() => {
          const mv = liveQuality.movement.score;
          const al = liveQuality.alignment.score;
          const li = liveQuality.lighting.score;
          let labelKey = '';
          let cls = '';
          if (!faceDetected) {
            labelKey = 'fl_no_face';
            cls = ' no-face';
          } else if (al !== null && al < 0.15) {
            labelKey = 'fl_align_critical';
            cls = ' error';
          } else if (al !== null && al < 0.33) {
            labelKey = 'fl_align_bad';
            cls = ' error';
          } else if (al !== null && al < 0.52) {
            labelKey = 'fl_align_off';
            cls = ' issue';
          } else if (li !== null && li < 0.15) {
            labelKey = 'fl_light_critical';
            cls = ' error';
          } else if (mv !== null && mv < 0.15) {
            labelKey = 'fl_move_extreme';
            cls = ' error';
          } else if (mv !== null && mv < 0.33) {
            labelKey = 'fl_move_bad';
            cls = ' error';
          } else if (li !== null && li < 0.3) {
            labelKey = 'fl_light_bad';
            cls = ' error';
          } else if (mv !== null && mv < 0.52) {
            labelKey = 'fl_move_slight';
            cls = ' issue';
          } else if (li !== null && li < 0.5) {
            labelKey = 'fl_light_dim';
            cls = ' issue';
          } else if (isMeasuring && mv !== null && mv < 0.6) {
            labelKey = 'fl_measuring_move';
            cls = ' issue';
          } else if (isMeasuring && li !== null && li < 0.65) {
            labelKey = 'fl_measuring_light';
            cls = ' issue';
          } else if (isMeasuring) {
            labelKey = 'fl_measuring_good';
            cls = ' ready';
          } else if (mv !== null && al !== null && li !== null && mv > 0.75 && al > 0.7 && li > 0.65) {
            labelKey = 'fl_ready_good';
            cls = ' ready';
          } else {
            labelKey = 'fl_ready';
            cls = ' ready';
          }
          return (
            <div className={`face-detected-label${cls}`}>
              <span className="fdi-dot" />
              {t(labelKey)}
            </div>
          );
        })()}
        {isMeasuring && (
          <div className="circular-progress" aria-label={`${t('cam_measuring')} ${progress}%`}>
            <svg width="96" height="96" viewBox="0 0 96 96">
              <circle cx="48" cy="48" r={RING_R} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="6" />
              <circle
                cx="48" cy="48" r={RING_R}
                fill="none"
                stroke="#5C7CFA"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={RING_C}
                strokeDashoffset={RING_C * (1 - progress / 100)}
                transform="rotate(-90 48 48)"
                style={{ transition: 'stroke-dashoffset 0.25s linear' }}
              />
            </svg>
            <span className="circular-progress-label">{progress}%</span>
          </div>
        )}
        {!isReady && !permissionError && (
          <div className="camera-init-overlay">
            <div className="skeleton" style={{ width: 120, height: 14, borderRadius: 7, margin: '0 auto 8px' }} />
            <small style={{ color: '#e0e7ff', opacity: 0.8 }}>{t('cam_init_overlay')}</small>
          </div>
        )}
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <p>{t(message)}</p>
      <div className="check-item" style={{ marginBottom: 10 }}>
        <div className="row between" style={{ marginBottom: 6 }}>
          <strong>{t('cam_quality_title')}</strong>
          <small>{t('cam_quality_sub')}</small>
        </div>
        {[
          { key: 'lighting', label: t('cam_q_lighting') },
          { key: 'movement', label: t('cam_q_movement') },
          { key: 'alignment', label: t('cam_q_alignment') },
        ].map((item) => {
          const q = liveQuality[item.key];
          const klass = q.status === 'good' ? 'good' : q.status === 'mid' ? 'mid' : q.status === 'bad' ? 'bad' : '';
          const statusText = q.status === 'good' ? t('cam_q_good') : q.status === 'mid' ? t('cam_q_mid') : q.status === 'bad' ? t('cam_q_bad') : t('cam_q_wait');
          return (
            <div key={item.key} className="check-item" style={{ marginBottom: 6 }}>
              <div className="row between">
                <strong>{item.label}</strong>
                <span className={`quality ${klass}`}>{statusText}{q.score !== null ? ` (${Math.round(q.score * 100)}%)` : ''}</span>
              </div>
              <small>{q.tip}</small>
            </div>
          );
        })}
      </div>
      <div className="progress-wrap" aria-hidden={!isMeasuring} style={{ opacity: isMeasuring ? 1 : 0.3 }}>
        <div className="progress" style={{ width: `${progress}%` }} />
      </div>
      {!isMeasuring && isReady && Object.values(liveQuality).some((q) => q.status === 'bad') && (
        <div className="quality-warning" role="alert">
          {t('cam_quality_warn')}
        </div>
      )}
      <div className="fab-spacer" />
    </div>
    {createPortal(
      <button
        onClick={startCapture}
        disabled={!isReady || isMeasuring || (faceMeshEnabled && !faceDetected)}
        className={`fab-measure${isMeasuring ? ' btn-measuring' : ''}`}
        title={faceMeshEnabled && !faceDetected ? t('cam_face_required') : undefined}
      >
        {isMeasuring
          ? t('cam_measuring')
          : !isReady
          ? t('cam_initializing')
          : (faceMeshEnabled && !faceDetected)
          ? t('cam_face_required')
          : t('cam_start')}
      </button>,
      document.body
    )}
  </>
  );
}

export default CameraView;
