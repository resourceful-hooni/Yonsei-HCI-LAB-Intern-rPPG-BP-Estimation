import { createContext, useContext, useEffect, useState } from 'react';
import translations from '../i18n/translations';

export const LANG_KEY = 'visi_lang';
const LangContext = createContext(null);

function initialLang() {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === 'ko' || saved === 'en') return saved;
  } catch (_) { /* storage unavailable */ }
  return 'ko';
}

export function LangProvider({ children }) {
  // Persisted so the choice survives reloads (previously it reset to Korean
  // on every page load, which made the EN toggle look like it "didn't work").
  const [lang, setLang] = useState(initialLang);

  useEffect(() => {
    try { localStorage.setItem(LANG_KEY, lang); } catch (_) { /* ignore */ }
    if (typeof document !== 'undefined') document.documentElement.lang = lang;
  }, [lang]);

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
