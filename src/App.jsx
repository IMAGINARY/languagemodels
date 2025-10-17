import React from 'react'
import ModelDemo from './components/ModelDemo.jsx'

export default function App() {
  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__emoji">🤖</span>
          <h1>AI Exhibits</h1>
        </div>
        <p className="subtitle">Exploring LLM concepts in the browser</p>
      </header>

      <main className="app__main">
        <section className="card">
          <h2>Word/Sentence Embeddings</h2>
          <p className="card__lead">
            Compute vector representations (embeddings) for text using
            transformers.js. Runs fully client-side.
          </p>
          <ModelDemo />
        </section>
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

