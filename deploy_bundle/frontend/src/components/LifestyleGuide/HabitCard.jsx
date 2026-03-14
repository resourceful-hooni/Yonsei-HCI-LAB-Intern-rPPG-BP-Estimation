function HabitCard({ item, onOpen }) {
  return (
    <div className="card habit-card">
      <h3>{item.icon} {item.title}</h3>
      <p>{item.description}</p>
      <button className="ghost link-btn" onClick={onOpen}>이 행동이 왜 필요한가요?</button>
    </div>
  );
}

export default HabitCard;
