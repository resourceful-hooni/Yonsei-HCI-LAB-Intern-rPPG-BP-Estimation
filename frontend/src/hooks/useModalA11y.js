import { useEffect, useRef } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

/**
 * Accessibility for dialog modals: move focus into the modal on open, trap Tab
 * within it, close on Escape, and restore focus to the previously-focused
 * element on close. Attach the returned ref to the modal container (which should
 * have tabIndex={-1} so it can receive focus when it has no focusable child).
 */
export default function useModalA11y(onClose) {
  const ref = useRef(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const node = ref.current;
    const prevActive = document.activeElement;

    const focusables = () =>
      node
        ? Array.from(node.querySelectorAll(FOCUSABLE)).filter((el) => el.offsetParent !== null)
        : [];

    const list = focusables();
    (list[0] || node)?.focus();

    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        if (onCloseRef.current) onCloseRef.current();
        return;
      }
      if (e.key === 'Tab' && node) {
        const f = focusables();
        if (f.length === 0) {
          e.preventDefault();
          return;
        }
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('keydown', onKeyDown, true);
      if (prevActive && typeof prevActive.focus === 'function') prevActive.focus();
    };
  }, []);

  return ref;
}
