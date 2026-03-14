import { useLang } from '../../contexts/LangContext';

function HabitCard({ item, onOpen }) {
  const { t } = useLang();
  return (
    <div className="card habit-card touchable">
      <h3>{item.icon} {item.title}</h3>
      <p>{item.description}</p>
      <button className="ghost link-btn" onClick={onOpen}>{t('lg_habit_why')}</button>
    </div>
  );
}

export default HabitCard;
