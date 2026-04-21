import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./i18n/locales/en.js";
import fr from "./i18n/locales/fr.js";
import de from "./i18n/locales/de.js";
import it from "./i18n/locales/it.js";

export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "fr", label: "Français" },
  { code: "de", label: "Deutsch" },
  { code: "it", label: "Italiano" },
];

const resources = {
  en: { translation: en },
  fr: { translation: fr },
  de: { translation: de },
  it: { translation: it },
};

const browserLanguage =
  typeof navigator === "undefined" ? "en" : navigator.language.split("-")[0];
const fallbackLanguage = "en";

const initialLanguage = SUPPORTED_LANGUAGES.some(
  ({ code }) => code === browserLanguage
)
  ? browserLanguage
  : fallbackLanguage;

i18n.use(initReactI18next).init({
  resources,
  lng: initialLanguage,
  fallbackLng: fallbackLanguage,
  interpolation: {
    escapeValue: false,
  },
});

console.assert(
  i18n.exists("menu.exhibits"),
  "[i18n] smoke test failed: translation key menu.exhibits is missing"
);
console.info("[i18n] initialized", {
  activeLanguage: i18n.language,
  exhibitsLabel: i18n.t("menu.exhibits"),
  supportedLanguages: SUPPORTED_LANGUAGES.map(({ code }) => code),
});

export default i18n;
