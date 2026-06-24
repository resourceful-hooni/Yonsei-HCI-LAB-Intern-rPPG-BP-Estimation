import { useLang } from '../../contexts/LangContext';
import Modal from '../common/Modal';

function WhyNeededModal({ item, onClose }) {
  const { t } = useLang();
  return (
    <Modal onClose={onClose} label={t('lg_why_title')}>
      <h3>{t('lg_why_title')}</h3>
      {item.reason_context && <p><strong>{item.reason_context}</strong></p>}
      <p>{item.detail || t('lg_why_default')}</p>
      <button onClick={onClose}>{t('common_close')}</button>
    </Modal>
  );
}

export default WhyNeededModal;
