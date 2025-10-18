// Minimal PCA via power iteration on the covariance operator.
// Works with small sets of high-dimensional vectors without extra deps.

// samples: Array of Float32Array (n x d)
// k: number of components (default 3)
export function computePCA(samples, k = 3, options = {}) {
  const n = samples.length;
  if (!n) throw new Error("computePCA: no samples");
  const d = samples[0].length;
  if (samples.some((v) => v.length !== d)) throw new Error("computePCA: inconsistent dimensions");
  const maxComponents = Math.min(k, Math.max(1, Math.min(d, n)));

  // Compute mean
  const mean = new Float32Array(d);
  for (let i = 0; i < n; i++) {
    const v = samples[i];
    for (let j = 0; j < d; j++) mean[j] += v[j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  // Centered data accessors (avoid materializing full X matrix)
  const centered = samples.map((v) => {
    const out = new Float32Array(d);
    for (let j = 0; j < d; j++) out[j] = v[j] - mean[j];
    return out;
  });

  const inv = n > 1 ? 1 / (n - 1) : 1;

  // A*v = (1/(n-1)) X^T X v computed lazily: X is (n x d) centered
  function mulA(v, out) {
    const tmp = new Float32Array(n); // X v  => length n
    for (let i = 0; i < n; i++) {
      const row = centered[i];
      let dot = 0;
      for (let j = 0; j < d; j++) dot += row[j] * v[j];
      tmp[i] = dot;
    }
    out.fill(0);
    for (let i = 0; i < n; i++) {
      const row = centered[i];
      const scale = tmp[i] * inv;
      for (let j = 0; j < d; j++) out[j] += row[j] * scale;
    }
  }

  // Utility ops
  function norm(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s) || 1e-12;
  }
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function axpy(a, x, y) {
    // y = y + a*x
    for (let i = 0; i < x.length; i++) y[i] += a * x[i];
  }
  function scale(v, s) {
    for (let i = 0; i < v.length; i++) v[i] *= s;
  }
  function copy(a, b) {
    for (let i = 0; i < a.length; i++) b[i] = a[i];
  }

  const maxIter = options.maxIter || 60;
  const tol = options.tol || 1e-6;

  const components = []; // array of Float32Array length d
  const explained = []; // eigenvalues

  // Find top-k components via power iteration with Gram-Schmidt orthogonalization
  for (let c = 0; c < maxComponents; c++) {
    let v = new Float32Array(d);
    // init random small vector
    for (let j = 0; j < d; j++) v[j] = (Math.random() - 0.5) * 2;
    // orthogonalize vs previous comps
    for (const q of components) {
      axpy(-dot(v, q), q, v);
    }
    scale(v, 1 / norm(v));

    const w = new Float32Array(d);
    let lastLambda = 0;
    for (let it = 0; it < maxIter; it++) {
      // w = A v
      mulA(v, w);
      // Deflate/orthogonalize against previous components
      for (const q of components) {
        axpy(-dot(w, q), q, w);
      }
      const vnorm = norm(w);
      if (vnorm === 0) break;
      scale(w, 1 / vnorm);

      // Rayleigh quotient as eigenvalue estimate
      const Av = new Float32Array(d);
      mulA(w, Av);
      const lambda = dot(w, Av);
      const delta = Math.abs(lambda - lastLambda);
      lastLambda = lambda;

      // convergence by direction change
      let diff = 0;
      for (let j = 0; j < d; j++) {
        const t = w[j] - v[j];
        diff += t * t;
      }
      copy(w, v);
      if (diff < tol * tol && delta < tol) break;
    }

    // finalize component c
    components.push(v);
    const Av = new Float32Array(d);
    mulA(v, Av);
    explained.push(dot(v, Av));
  }

  // Project centered samples into component space (n x maxComponents)
  const coords = centered.map((row) => {
    const out = new Float32Array(components.length);
    for (let c = 0; c < components.length; c++) {
      out[c] = dot(row, components[c]);
    }
    return out;
  });

  return {
    mean,
    components, // Array of length-k Float32Array (d each)
    explainedVariance: explained,
    coords, // Array of n Float32Array (k each)
  };
}

