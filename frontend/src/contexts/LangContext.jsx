import { createContext, useContext, useState } from 'react';
import translations from '../i18n/translations';

const LangContext = createContext(null);

export function LangProvider({ children }) {
  const [lang, setLang] = useState('ko');

  /**
   * Translate a key, optionally substituting {varName} placeholders.
   * Falls back to Korean, then to the raw key string.
   */
  const t = (key, vars = {}) => {
    let str =
      translations[lang]?.[key] ??
      translations.ko?.[key] ??
      key;
    Object.entries(vars).forEach(([k, v]) => {
      str = str.replace(`{${k}}`, String(v));
    });
    return str;
  };

  const toggle = () => setLang((prev) => (prev === 'ko' ? 'en' : 'ko'));

  return (
    <LangContext.Provider value={{ lang, t, toggle }}>
      {children}
    </LangContext.Provider>
  );
}

export const useLang = () => useContext(LangContext);
