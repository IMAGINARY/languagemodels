import React, { useCallback, useMemo, useRef, useState } from "react";

// Simple in-memory cache for the pipeline during the session
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

const logPrefix = "[transformers-demo]";

// As a safety net, intercept all fetches and redirect any relative
// '/Xenova/...' paths to our local '/models/Xenova/...' mirror.
if (typeof window !== 'undefined' && !window.__xfetchPatched) {
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    let inputUrl = typeof input === 'string' ? input : (input?.url || '');
    let url;
    try { url = new URL(inputUrl, window.location.origin); } catch { url = null; }

    // Build list of candidate URLs to try if the first returns HTML/404
    const candidates = [];
    if (url) {
      // Default
      candidates.push(url.toString());

      // If transformers requested '/Xenova/...', rewrite to local mirror under '/models'
      if (url.origin === window.location.origin && url.pathname.startsWith('/Xenova/')) {
        const local = new URL(url.toString());
        local.pathname = url.pathname.replace(/^\/Xenova\//, '/models/Xenova/');
        candidates.unshift(local.toString()); // prefer local mirror first
      }

      // If a local model path without '/resolve/main', add a candidate with it
      if (url.origin === window.location.origin && url.pathname.startsWith('/models/Xenova/') && !url.pathname.includes('/resolve/main/')) {
        const withResolve = new URL(url.toString());
        withResolve.pathname = withResolve.pathname.replace('/models/Xenova/', '/models/Xenova/').replace(/^(.*?\/Xenova\/[^/]+\/)/, '$1resolve/main/');
        candidates.push(withResolve.toString());

        // If requesting quantized model, also try non-quantized filename
        if (/\/onnx\/model_quantized\.onnx$/i.test(withResolve.pathname)) {
          const nonq = new URL(withResolve.toString());
          nonq.pathname = nonq.pathname.replace('model_quantized.onnx', 'model.onnx');
          candidates.push(nonq.toString());
        }
      }
    } else {
      candidates.push(inputUrl || String(input));
    }

    // Try candidates in order until we get a non-HTML successful response
    for (let i = 0; i < candidates.length; i++) {
      const finalUrl = candidates[i];
      console.log(logPrefix, i === 0 ? 'fetch (win) ->' : 'fetch (win,alt) ->', finalUrl);
      const res = await originalFetch(finalUrl, init);
      const ct = res.headers.get('content-type') || '';
      console.log(logPrefix, 'fetch (win) <-', res.status, ct, 'for', finalUrl);
      const isHtml = /text\/html/i.test(ct);
      const isOk = res.ok && !isHtml;
      if (isOk) return res;
      // If last candidate, consider JSON stubs
      const isJson = /\.json($|\?)/i.test(finalUrl);
      const isLocalModelsJson = finalUrl.includes('/models/Xenova/') && isJson;
      if (i === candidates.length - 1 && isLocalModelsJson) {
        const text = await res.clone().text().catch(() => '');
        if (!res.ok || isHtml || /^\s*<!doctype/i.test(text) || text.trim() === '') {
          console.log(logPrefix, 'stub JSON served for', finalUrl);
          return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
        }
      }
      // Otherwise continue to next candidate
    }

    // Fallback: last fetch
    const last = candidates[candidates.length - 1];
    return originalFetch(last, init);
  };
  window.__xfetchPatched = true;
}

// Dynamically import transformers after fetch is patched so our overrides apply
let TF = null;
async function getTransformers() {
  if (!TF) {
    TF = await import("@xenova/transformers");
    const { env } = TF;
    // Offline-first: serve ONNX runtime and models locally
    env.allowLocalModels = true;
    env.localModelPath = "/models"; // served from public/models
    env.backends.onnx.wasm.wasmPaths = "/transformers/"; // served from public/transformers
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.proxy = false; // keep in main thread for simplicity/offline
    // Avoid using any previously cached (possibly bad) files
    env.useBrowserCache = false;

    // Log each fetch to aid debugging
    const origEnvFetch = typeof env.fetch === 'function' ? env.fetch : fetch.bind(window);
    env.fetch = async (url, options) => {
      let u = url;
      if (typeof u === 'string' && !/^https?:\/\//i.test(u)) {
        // Map any Xenova path to our local models mirror
        if (u.startsWith('/Xenova/')) {
          u = u.replace(/^\/Xenova\//, '/models/Xenova/');
        } else if (u.startsWith('Xenova/')) {
          u = '/models/' + u;
        } else if (u.includes('Xenova/')) {
          u = '/models/' + u.slice(u.indexOf('Xenova/'));
        }
      }
      console.log(logPrefix, 'fetch (env) ->', u);
      const res = await origEnvFetch(u, options);
      console.log(
        logPrefix,
        "fetch (env) <-",
        res.status,
        res.headers.get("content-type"),
        "for",
        u
      );
      return res;
    };
  }
  return TF;
}

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Model not loaded");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState(
    "Transformers are really cool for embeddings!"
  );
  const [events, setEvents] = useState([]);
  const abortRef = useRef(null);

  const canRun = useMemo(
    () => status !== "loading" && status !== "running",
    [status]
  );

  const verifyLocalModel = useCallback(async () => {
    const base = `/models/${LOCAL_MODEL_ID}/resolve/main`;
    const required = [
      `${base}/config.json`,
      `${base}/tokenizer.json`,
      `${base}/tokenizer_config.json`,
      `${base}/onnx/model.onnx`,
    ];
    const optional = [
      `${base}/special_tokens_map.json`,
      `${base}/preprocessor_config.json`,
    ];
    const results = [];
    for (const url of [...required, ...optional]) {
      try {
        const res = await fetch(url, { method: 'GET' });
        const ct = res.headers.get('content-type') || '';
        const ok = res.ok && !/text\/html/i.test(ct);
        const isOptional = optional.includes(url);
        results.push({ url, ok, status: res.status, ct, optional: isOptional });
      } catch (e) {
        const isOptional = optional.includes(url);
        results.push({ url, ok: false, status: 0, ct: '', err: String(e), optional: isOptional });
      }
    }
    return results;
  }, []);

  const loadModel = useCallback(async () => {
    try {
      setError(null);
      setStatus("loading");
      setMessage("Loading model (first time may take a while) …");
      setEvents((e) =>
        [{ t: Date.now(), m: "init loadModel" }, ...e].slice(0, 500)
      );
      console.log(logPrefix, "loadModel: start");

      // Verify local files exist before attempting pipeline
      const checks = await verifyLocalModel();
      const missing = checks.filter((c) => !c.ok && !c.optional);
      if (missing.length) {
        const lines = missing.map(
          (m) => `missing or invalid (${m.status} ${m.ct || ''}): ${m.url}`
        );
        const msg = `Local model files not found or invalid.\n- ${lines.join('\n- ')}`;
        console.error(logPrefix, msg);
        setStatus('error');
        setError(msg);
        setMessage('Local model missing');
        setEvents((e) => [{ t: Date.now(), m: msg }, ...e].slice(0, 50));
        return;
      }
      const missingOptional = checks.filter((c) => !c.ok && c.optional);
      if (missingOptional.length) {
        const lines = missingOptional.map(
          (m) => `optional missing (${m.status} ${m.ct || ''}): ${m.url}`
        );
        setEvents((e) => [{ t: Date.now(), m: lines.join('\n') }, ...e].slice(0, 50));
      }

      // Model: Small sentence embedding model
      // Note: transformers.js caches model files in IndexedDB by default
      const { pipeline } = await getTransformers();
      extractor = await pipeline(
        "feature-extraction",
        LOCAL_MODEL_ID,
        {
          revision: 'main',
          local_files_only: true,
          progress_callback: (p) => {
            setEvents((e) =>
              [{ t: Date.now(), m: JSON.stringify(p) }, ...e].slice(0, 50)
            );
            if (p?.status && p?.name) {
              setMessage(`${p.status}: ${p.name}`);
            }
            // Also log to console for Network tab correlation
            console.log(logPrefix, "progress", p);
          },
        }
      );

      setStatus("ready");
      setMessage("Model ready");
      console.log(logPrefix, "loadModel: ready");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to load model. Check console for details.");
      setEvents((ev) =>
        [{ t: Date.now(), m: `error: ${e?.message || e}` }, ...ev].slice(0, 50)
      );
    }
  }, []);

  const runEmbedding = useCallback(async () => {
    try {
      setError(null);
      setStatus("running");
      setMessage("Computing embedding …");

      if (!extractor) {
        // Lazy-load if user skips the explicit load step
        const checks = await verifyLocalModel();
        const missing = checks.filter((c) => !c.ok && !c.optional);
        if (missing.length) {
          const lines = missing.map(
            (m) => `missing or invalid (${m.status} ${m.ct || ''}): ${m.url}`
          );
          const msg = `Local model files not found or invalid.\n- ${lines.join('\n- ')}`;
          console.error(logPrefix, msg);
          setStatus('error');
          setError(msg);
          setMessage('Local model missing');
          setEvents((e) => [{ t: Date.now(), m: msg }, ...e].slice(0, 50));
          return;
        }
        const missingOptional = checks.filter((c) => !c.ok && c.optional);
        if (missingOptional.length) {
          const lines = missingOptional.map(
            (m) => `optional missing (${m.status} ${m.ct || ''}): ${m.url}`
          );
          setEvents((e) => [{ t: Date.now(), m: lines.join('\n') }, ...e].slice(0, 50));
        }
        const { pipeline } = await getTransformers();
        extractor = await pipeline(
          "feature-extraction",
          LOCAL_MODEL_ID,
          {
            revision: 'main',
            local_files_only: true,
            progress_callback: (p) => {
              setEvents((e) =>
                [{ t: Date.now(), m: JSON.stringify(p) }, ...e].slice(0, 50)
              );
              if (p?.status && p?.name) {
                setMessage(`${p.status}: ${p.name}`);
              }
              console.log(logPrefix, "progress", p);
            },
          }
        );
      }

      const output = await extractor(text, {
        pooling: "mean",
        normalize: true,
      });
      const data = output?.data ?? [];
      const dims = output?.dims ?? [data.length];

      const preview = Array.from(data).slice(0, 8);
      setVectorInfo({
        length: data.length,
        dims: dims.join("×"),
        preview,
      });

      setStatus("ready");
      setMessage("Done");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to compute embedding. Check console for details.");
      setEvents((ev) =>
        [{ t: Date.now(), m: `error: ${e?.message || e}` }, ...ev].slice(0, 50)
      );
    }
  }, [text]);

  const cancel = useCallback(() => {
    try {
      abortRef.current?.abort?.();
      setStatus("idle");
      setMessage("Cancelled");
    } catch {}
  }, []);

  return (
    <div className="panel">
      <div className="row">
        <textarea
          className="input"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type some text to embed…"
        />
      </div>

      <div className="row row--actions">
        <button className="btn" onClick={loadModel} disabled={!canRun}>
          {status === "loading" ? "Loading…" : "Load model"}
        </button>
        <button
          className="btn btn--primary"
          onClick={runEmbedding}
          disabled={!canRun}
        >
          {status === "running" ? "Running…" : "Compute embedding"}
        </button>
        <span className={`status status--${status}`}>{message}</span>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {vectorInfo && (
        <div className="result">
          <div className="result__meta">
            <span>
              <strong>Dims:</strong> {vectorInfo.dims}
            </span>
            <span>
              <strong>Length:</strong> {vectorInfo.length}
            </span>
          </div>
          <div className="result__preview">
            <strong>Preview:</strong>
            <code>
              [{" "}
              {vectorInfo.preview.map((v, i) => (
                <span key={i}>
                  {i ? ", " : ""}
                  {v.toFixed(4)}
                </span>
              ))}{" "}
              … ]
            </code>
          </div>
        </div>
      )}

      {events.length > 0 && (
        <div className="result">
          <div className="result__meta">
            <strong>Events</strong>
            <span className="status">(latest first)</span>
          </div>
          <div className="result__preview">
            <code>
              {events.slice(0, 8).map((e, i) => (
                <div key={i}>
                  • {new Date(e.t).toLocaleTimeString()} — {e.m}
                </div>
              ))}
            </code>
          </div>
        </div>
      )}
    </div>
  );
}
