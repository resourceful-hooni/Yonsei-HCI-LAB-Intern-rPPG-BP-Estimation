import { createPortal } from 'react-dom';
import useModalA11y from '../../hooks/useModalA11y';

/**
 * Shared accessible modal: backdrop-click and Escape close it, focus is trapped
 * inside while open and restored to the trigger on close (see useModalA11y).
 * Replaces the ad-hoc createPortal blocks that previously had no focus handling.
 */
function Modal({ onClose, label, className = '', children }) {
  const ref = useModalA11y(onClose);
  return createPortal(
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && onClose) onClose();
      }}
    >
      <div
        className={`modal ${className}`.trim()}
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body
  );
}

export default Modal;
