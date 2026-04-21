import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import ModelDemo1 from "./components/ModelDemo1.jsx";
import ModelDemo2 from "./components/ModelDemo2.jsx";
import translateIcon from "./img/translate.svg";
import { SUPPORTED_LANGUAGES } from "./i18n.js";

export default function App() {
  const [view, setView] = useState("menu"); // 'menu' | 'model1' | 'model2'
  const { i18n, t } = useTranslation();
  const languageMenuRef = useRef(null);
  const [selectedLanguage, setSelectedLanguage] = useState(
    i18n.resolvedLanguage || i18n.language || "en"
  );

  useEffect(() => {
    const handlePointerDown = (event) => {
      if (!languageMenuRef.current?.contains(event.target)) {
        languageMenuRef.current?.removeAttribute("open");
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, []);

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__header-main">
          <div>
            <div className="brand">
              <span className="brand__emoji">🤖</span>
              <h1>{t("app.title")}</h1>
            </div>
            <p className="subtitle">{t("app.subtitle")}</p>
          </div>

          {view === "menu" ? (
            <details className="language-menu" ref={languageMenuRef}>
              <summary
                className="language-menu__trigger"
                aria-label={t("menu.language")}
                title={t("menu.language")}
              >
                <img src={translateIcon} alt="" aria-hidden="true" />
              </summary>

              <div
                className="language-menu__dropdown"
                role="menu"
                aria-label={t("menu.language")}
              >
                {SUPPORTED_LANGUAGES.map(({ code, label }) => (
                  <button
                    key={code}
                    type="button"
                    className={`language-menu__item${
                      selectedLanguage === code
                        ? " language-menu__item--active"
                        : ""
                    }`}
                    onClick={() => {
                      setSelectedLanguage(code);
                      i18n.changeLanguage(code);
                      languageMenuRef.current?.removeAttribute("open");
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </details>
          ) : null}
        </div>
      </header>

      <main className="app__main">
        {view === "menu" && (
          <section className="card">
            <h2>{t("menu.exhibits")}</h2>
            <p className="card__lead">{t("menu.pickDemo")}</p>
            <ul className="menu">
              <li>
                <button
                  className="btn btn--primary"
                  onClick={() => setView("model1")}
                >
                  {t("menu.embeddingsExplorer")}
                </button>
              </li>

              <li>
                <button
                  className="btn btn--primary"
                  onClick={() => setView("model2")}
                >
                  {t("menu.tokenizationAttention")}
                </button>
              </li>
            </ul>
          </section>
        )}

        {view === "model1" && (
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>{t("menu.embeddingsExplorer")}</h2>
              <button className="btn" onClick={() => setView("menu")}>
                ← {t("menu.backToMenu")}
              </button>
            </div>
            <p className="card__lead">{t("model1.lead")}</p>
            <ModelDemo1 language={selectedLanguage} />
          </section>
        )}

        {view === "model2" && (
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>{t("menu.tokenizationAttention")}</h2>
              <button className="btn" onClick={() => setView("menu")}>
                ← {t("menu.backToMenu")}
              </button>
            </div>
            <p className="card__lead">{t("model2.lead")}</p>
            <ModelDemo2 language={selectedLanguage} />
          </section>
        )}

      </main>

      <footer className="app__footer">{t("app.footer")}</footer>
    </div>
  );
}
