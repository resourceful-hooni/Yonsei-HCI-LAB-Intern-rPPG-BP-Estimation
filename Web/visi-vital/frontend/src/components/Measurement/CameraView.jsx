import { useEffect, useRef, useState } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { processMeasurement, startMeasurement } from '../../services/apiService';

const TOTAL_FRAMES = 300;
const FPS = 30;
const CAPTURE_WIDTH = 320;
const JPEG_QUALITY = 0.55;

function CameraView({ userId, onResult }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const timerRef = useRef(null);
  const rafRef = useRef(null);
  const faceMeshRef = useRef(null);
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

  const [isReady, setIsReady] = useState(false);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceMeshEnabled, setFaceMeshEnabled] = useState(true);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('얼굴을 화면 중앙에 위치시켜주세요');
  const [liveQuality, setLiveQuality] = useState({
    lighting: { score: null, status: '대기', tip: '조명을 확인 중입니다.' },
    movement: { score: null, status: '대기', tip: '움직임을 확인 중입니다.' },
    alignment: { score: null, status: '대기', tip: '얼굴 정렬을 확인 중입니다.' },
  });

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));

  const statusFromScore = (score) => {
    if (score === null || Number.isNaN(Number(score))) return '대기';
    const v = Number(score);
    if (v >= 0.75) return '좋음';
    if (v >= 0.55) return '보통';
    return '개선 필요';
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
    let unmounted = false;

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
          setMessage('얼굴이 감지되었습니다. 편안한 자세로 움직이지 말고 유지해주세요.');
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
              setMessage('얼굴이 감지되었습니다. 편안한 자세로 움직이지 말고 유지해주세요.');
            } else {
              setMessage('얼굴이 감지되지 않았으니 얼굴을 화면 중앙에 위치시켜주세요.');
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
            const liveCtx = liveCanvas?.getContext('2d');
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
                tip: '얼굴 정면에 균일한 빛이 들어오도록 조명을 조정해보세요.',
              },
              movement: {
                score: movementScore,
                status: statusFromScore(movementScore),
                tip: '측정 중에는 시선과 고개를 가능한 고정해 주세요.',
              },
              alignment: {
                score: alignmentScore,
                status: statusFromScore(alignmentScore),
                tip: '얼굴을 화면 중앙에 두고 턱선이 프레임 안에 들어오게 맞춰주세요.',
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

            ctx.strokeStyle = '#5C7CFA';
            ctx.lineWidth = 1;
            points.forEach((pt) => {
              const x = overlay.width - (pt.x * videoW * scale + offsetX);
              const y = pt.y * videoH * scale + offsetY;
              ctx.beginPath();
              ctx.arc(x, y, 1, 0, Math.PI * 2);
              ctx.stroke();
            });

            // 중앙 상단 텍스트 (항상 정방향)
            ctx.save();
            if (typeof ctx.resetTransform === 'function') {
              ctx.resetTransform();
            } else {
              ctx.setTransform(1, 0, 0, 1, 0, 0);
            }

            const text = '얼굴이 감지되었습니다.';
            ctx.font = 'bold 16px sans-serif';
            const metrics = ctx.measureText(text);
            const padX = 12;
            const boxW = metrics.width + padX * 2;
            const boxH = 30;
            const boxX = (overlay.width - boxW) / 2;
            const boxY = 14;

            ctx.fillStyle = 'rgba(15, 23, 42, 0.45)';
            ctx.fillRect(boxX, boxY, boxW, boxH);
            ctx.fillStyle = '#e0e7ff';
            ctx.fillText(text, boxX + padX, boxY + 20);
            ctx.restore();

          }
          if (!hasFace) {
            prevFaceCenterRef.current = null;
            latestFaceMetricsRef.current = { hasFace: false, movementScore: null, alignmentScore: null };
            setLiveQuality((prev) => ({
              lighting: prev.lighting,
              movement: {
                score: null,
                status: '대기',
                tip: '얼굴이 감지되면 움직임 품질을 실시간으로 표시합니다.',
              },
              alignment: {
                score: null,
                status: '대기',
                tip: '얼굴이 감지되면 정렬 품질을 실시간으로 표시합니다.',
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
            setMessage('얼굴 가이드 로딩에 문제가 있어 측정을 계속 진행합니다.');
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

    init().catch(() => setMessage('카메라 권한을 확인해주세요.'));

    return () => {
      unmounted = true;
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
      setMessage('얼굴이 감지되지 않았으니 얼굴을 화면 중앙에 위치시켜주세요.');
      return;
    }

    setIsMeasuring(true);
    setMessage('측정 중... 움직이지 마시고 편안하게 유지해주세요');
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
          const ctx = canvas?.getContext('2d');
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
          timerRef.current = null;
          setMessage('분석 중입니다...');

          try {
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

            const res = await processMeasurement(measurementId, frames, FPS, qualityMetrics);
            onResult(res.data);
            setMessage('밝은 곳에서 측정하면 더 정확해요');
          } catch (err) {
            if ((err.message || '').includes('Payload Too Large') || (err.message || '').includes('전송 데이터가 너무 큽니다')) {
              setMessage('전송 용량이 커서 실패했어요. 자동 압축 설정을 적용했으니 다시 시도해주세요.');
            } else {
              setMessage(err.message || '분석 중 오류가 발생했습니다. 다시 시도해주세요.');
            }
          } finally {
            setIsMeasuring(false);
          }
        }
      }, Math.round(1000 / FPS));
    } catch (err) {
      if ((err.message || '').includes('Payload Too Large') || (err.message || '').includes('전송 데이터가 너무 큽니다')) {
        setMessage('전송 용량이 커서 실패했어요. 자동 압축 설정을 적용했으니 다시 시도해주세요.');
      } else {
        setMessage(err.message || '측정 중 오류가 발생했습니다.');
      }
      setIsMeasuring(false);
    }
  };

  return (
    <div className="card">
      <div className="camera-wrap">
        <video ref={videoRef} className="camera" muted playsInline />
        <canvas ref={overlayRef} className="overlay" />
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <p>{message}</p>
      <div className="check-item" style={{ marginBottom: 10 }}>
        <div className="row between" style={{ marginBottom: 6 }}>
          <strong>실시간 측정 품질 체크리스트</strong>
          <small>카메라 실측</small>
        </div>
        {[
          { key: 'lighting', label: '조명' },
          { key: 'movement', label: '움직임' },
          { key: 'alignment', label: '얼굴정렬' },
        ].map((item) => {
          const q = liveQuality[item.key];
          const klass = q.status === '좋음' ? 'good' : q.status === '보통' ? 'mid' : q.status === '개선 필요' ? 'bad' : '';
          return (
            <div key={item.key} className="check-item" style={{ marginBottom: 6 }}>
              <div className="row between">
                <strong>{item.label}</strong>
                <span className={`quality ${klass}`}>{q.status}{q.score !== null ? ` (${Math.round(q.score * 100)}%)` : ''}</span>
              </div>
              <small>{q.tip}</small>
            </div>
          );
        })}
      </div>
      <div className="progress-wrap">
        <div className="progress" style={{ width: `${progress}%` }} />
      </div>
      <button onClick={startCapture} disabled={!isReady || isMeasuring}>
        {isMeasuring ? '측정 중...' : '10초 측정 시작'}
      </button>
    </div>
  );
}

export default CameraView;
