import { createPortal } from 'react-dom';
import { useLang } from '../../contexts/LangContext';

function WhyNeededModal({ item, onClose }) {
  const { t } = useLang();
  return createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t('lg_why_title')}</h3>
        {item.reason_context && <p><strong>{item.reason_context}</strong></p>}
        <p>{item.detail || t('lg_why_default')}</p>
        <button onClick={onClose}>{t('common_close')}</button>
      </div>
    </div>,
    document.body
  );
}

export default WhyNeededModal;
