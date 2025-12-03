import React, { useState } from "react";
import ModelDemo1 from "./components/ModelDemo1.jsx";
import ModelDemo2 from "./components/ModelDemo2.jsx";
import ModelDemo3 from "./components/ModelDemo3.jsx";

export default function App() {
  const [view, setView] = useState("menu"); // 'menu' | 'model1' | 'model2' | 'model3'

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__emoji">🤖</span>
          <h1>AI Exhibits. Large Language Models</h1>
        </div>
        <p className="subtitle">Exploring LLM concepts in the browser</p>
      </header>

      <main className="app__main">
        {view === "menu" && (
          <section className="card">
            <h2>Exhibits</h2>
            <p className="card__lead">Pick a demo to explore</p>
            <ul className="menu">
              <li>
                <button
                  className="btn btn--primary"
                  onClick={() => setView("model1")}
                >
                  Word/Sentence embeddings
                </button>
              </li>
              <li>
                <button className="btn" onClick={() => setView("model2")}>
                  Tokenization and attention
                </button>
              </li>
              <li>
                <button className="btn" onClick={() => setView("model3")}>
                  Token similarity (vector KNN)
                </button>
              </li>
            </ul>
          </section>
        )}

        {view === "model1" && (
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Word/Sentence Embeddings</h2>
              <button className="btn" onClick={() => setView("menu")}>
                ← Back to menu
              </button>
            </div>
            <p className="card__lead">
              Compute vector representations (embeddings) for text. Add at least
              3 vectors to dataset to visualize.
            </p>
            <ModelDemo1 />
          </section>
        )}

        {view === "model2" && (
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Tokenization and attention</h2>
              <button className="btn" onClick={() => setView("menu")}>
                ← Back to menu
              </button>
            </div>
            <p className="card__lead">
              Explore how tokenization works and visualize attention maps.
            </p>
            <ModelDemo2 />
          </section>
        )}

        {view === "model3" && (
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Token similarity (vector KNN)</h2>
              <button className="btn" onClick={() => setView("menu")}>
                ← Back to menu
              </button>
            </div>
            <p className="card__lead">
              Enter a word or short phrase, find the 5 nearest tokens, and view a PCA of the query plus neighbors.
            </p>
            <ModelDemo3 />
          </section>
        )}
      </main>

      <footer className="app__footer">IMAGINARY, 2025</footer>
    </div>
  );
}
