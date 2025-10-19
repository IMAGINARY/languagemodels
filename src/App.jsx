import React, { useState } from 'react'
import ModelDemo1 from './components/ModelDemo1.jsx'
import ModelDemo2 from './components/ModelDemo2.jsx'

export default function App() {
  const [view, setView] = useState('menu') // 'menu' | 'model1' | 'model2'

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
        {view === 'menu' && (
          <section className="card">
            <h2>Exhibits</h2>
            <p className="card__lead">Pick a demo to explore</p>
            <ul className="menu">
              <li>
                <button className="btn btn--primary" onClick={() => setView('model1')}>
                  Word/Sentence embeddings — Model 1
                </button>
              </li>
              <li>
                <button className="btn" onClick={() => setView('model2')}>
                  Word/Sentence embeddings — Model 2 (placeholder)
                </button>
              </li>
            </ul>
          </section>
        )}

        {view === 'model1' && (
          <section className="card">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>Word/Sentence Embeddings</h2>
              <button className="btn" onClick={() => setView('menu')}>← Back to menu</button>
            </div>
            <p className="card__lead">
              Compute vector representations (embeddings) for text using transformers.js. Runs fully client-side.
            </p>
            <ModelDemo1 />
          </section>
        )}

        {view === 'model2' && (
          <section className="card">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>Word/Sentence Embeddings — Model 2</h2>
              <button className="btn" onClick={() => setView('menu')}>← Back to menu</button>
            </div>
            <p className="card__lead">
              Placeholder: identical to Model 1 for now. We will modify it later.
            </p>
            <ModelDemo2 />
          </section>
        )}
      </main>

      <footer className="app__footer">
        <span>
          Built with React + <code>@xenova/transformers</code>
        </span>
        <a href="https://xenova.github.io/transformers.js/" target="_blank" rel="noreferrer">
          transformers.js docs
        </a>
      </footer>
    </div>
  )
}
