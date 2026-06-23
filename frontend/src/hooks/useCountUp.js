import { useEffect, useRef, useState } from 'react';

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Animate a number from 0 → target with an ease-out curve.
 * Returns the current (rounded) value. Honors prefers-reduced-motion
 * (jumps straight to the target).
 */
export default function useCountUp(target, { duration = 900, decimals = 0 } = {}) {
  const end = Number(target) || 0;
  const [value, setValue] = useState(end);
  const rafRef = useRef(0);

  useEffect(() => {
    if (prefersReducedMotion() || !end) {
      setValue(end);
      return undefined;
    }
    let start = null;
    const factor = Math.pow(10, decimals);
    const tick = (ts) => {
      if (start === null) start = ts;
      const p = Math.min(1, (ts - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      setValue(Math.round(end * eased * factor) / factor);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [end, duration, decimals]);

  return value;
}
