function WhyNeededModal({ item, onClose }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>현재 단계에서의 의미</h3>
        {item.reason_context && <p><strong>{item.reason_context}</strong></p>}
        <p>{item.detail || '최근 혈압 변동성이 유지될 경우, 심혈관 건강 관리에 지속적인 관심이 필요할 수 있습니다. 지금은 생활습관 개선을 통해 변동성을 줄여나가는 단계로 이해하시면 됩니다.'}</p>
        <button onClick={onClose}>닫기</button>
      </div>
    </div>
  );
}

export default WhyNeededModal;
