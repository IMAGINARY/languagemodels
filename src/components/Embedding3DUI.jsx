import React, { useEffect, useMemo, useState } from "react";
import { PCA as PCAClass } from "ml-pca";
import Embedding3D from "./Embedding3D.jsx";

const EPS = 1e-10;

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

function subtractProjection(vector, basis) {
  const out = Float64Array.from(vector);
  for (const basisVector of basis) {
    const amount = dot(out, basisVector);
    for (let i = 0; i < out.length; i++) out[i] -= amount * basisVector[i];
  }
  return out;
}

function buildBasis(seedVectors) {
  const basis = [];
  for (const vector of seedVectors) {
    const residual = subtractProjection(vector, basis);
    const residualNorm = norm(residual);
    if (residualNorm <= EPS) continue;

    const unit = new Float64Array(residual.length);
    for (let i = 0; i < residual.length; i++) unit[i] = residual[i] / residualNorm;
    basis.push(unit);
  }
  return basis;
}

function projectIsometrically(items) {
  if (!items.length) {
    return {
      coords: [],
      modeLabel: "Isometric",
      subtitle: "No vectors",
    };
  }

  const basis = buildBasis(items.slice(0, 3).map((item) => item.vector));
  const coords = items.map((item) => {
    const vector = item.vector;
    const z = basis[0] ? dot(vector, basis[0]) : 0;
    const y = basis[1] ? dot(vector, basis[1]) : 0;
    const x = basis[2] ? dot(vector, basis[2]) : 0;
    return [x, y, z];
  });

  return {
    coords,
    modeLabel: "Isometric",
    subtitle:
      items.length >= 3
        ? "Basis from the first three vectors"
        : `Basis from the first ${items.length} vector${items.length === 1 ? "" : "s"}`,
  };
}

function projectWithPCA(items) {
  if (items.length < 2) {
    return {
      coords: [],
      modeLabel: "PCA",
      subtitle: "Add at least two vectors for PCA",
      explainedVariance: null,
    };
  }

  const matrix = items.map((item) => Array.from(item.vector));
  const pca = new PCAClass(matrix, { center: true, scale: false });
  const projection = pca.predict(matrix, { nComponents: 3 });
  const coords = projection.to2DArray ? projection.to2DArray() : projection;

  return {
    coords,
    modeLabel: "PCA",
    subtitle: "Top 3 principal components",
    explainedVariance: pca.getExplainedVariance()?.slice(0, 3) ?? null,
  };
}

export default function Embedding3DUI({
  items = [],
  height = 420,
  title = "3D Embedding Viewer",
  onClearItems,
  onDeleteItem,
}) {
  const [projectionMode, setProjectionMode] = useState("isometric");
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const normalizedItems = useMemo(
    () =>
      items.map((item, index) => ({
        label: item.label || `#${index + 1}`,
        vector: item.vector,
      })),
    [items]
  );

  const projection = useMemo(() => {
    if (projectionMode === "pca") return projectWithPCA(normalizedItems);
    return projectIsometrically(normalizedItems);
  }, [normalizedItems, projectionMode]);

  useEffect(() => {
    if (selectedIndex !== null && selectedIndex >= normalizedItems.length) {
      setSelectedIndex(null);
    }
    if (hoveredIndex !== null && hoveredIndex >= normalizedItems.length) {
      setHoveredIndex(null);
    }
  }, [hoveredIndex, normalizedItems.length, selectedIndex]);

  const points = useMemo(() => {
    const activeIndex = hoveredIndex ?? selectedIndex;
    return normalizedItems.map((item, index) => {
      const coords = projection.coords[index] ?? [0, 0, 0];
      const isActive = activeIndex === index;
      const hasActive = activeIndex !== null;
      return {
        x: coords[0] ?? 0,
        y: coords[1] ?? 0,
        z: coords[2] ?? 0,
        label: item.label,
        color: isActive
          ? "#f59e0b"
          : hasActive
            ? "#4b5563"
            : "#6ee7ff",
      };
    });
  }, [hoveredIndex, normalizedItems, projection.coords, selectedIndex]);

  const explainedLabel = useMemo(() => {
    if (!projection.explainedVariance?.length) return null;
    return projection.explainedVariance
      .map((value, index) => `PC${index + 1}: ${value.toFixed(3)}`)
      .join(", ");
  }, [projection.explainedVariance]);

  return (
    <div className="embedding3d-ui">
      <div className="embedding3d-ui__sidebar">
        <div className="embedding3d-ui__toolbar">
          <div className="embedding3d-ui__toolbar-head">
            <div>
              <h3 className="embedding3d-ui__title">{title}</h3>
              <p className="embedding3d-ui__subtitle">{normalizedItems.length} vectors</p>
            </div>
            <button
              type="button"
              className="btn embedding3d-ui__clear"
              onClick={() => onClearItems?.()}
              disabled={!normalizedItems.length}
            >
              Clear all
            </button>
          </div>
          <label className="embedding3d-ui__control">
            <span>Projection</span>
            <select
              className="input embedding3d-ui__select"
              value={projectionMode}
              onChange={(e) => setProjectionMode(e.target.value)}
            >
              <option value="isometric">Isometric</option>
              <option value="pca">PCA</option>
            </select>
          </label>
        </div>

        <div className="embedding3d-ui__meta">
          <span>{projection.modeLabel}</span>
          <span>{projection.subtitle}</span>
          {explainedLabel ? <span>{explainedLabel}</span> : null}
        </div>

        <div className="embedding3d-ui__list" role="list">
          {normalizedItems.map((item, index) => {
            const isSelected = selectedIndex === index;
            const isHovered = hoveredIndex === index;
            return (
              <div
                key={`${item.label}-${index}`}
                className={`embedding3d-ui__item${isSelected || isHovered ? " embedding3d-ui__item--active" : ""}`}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex((current) => (current === index ? null : current))}
              >
                <button
                  type="button"
                  className="embedding3d-ui__item-main"
                  onClick={() =>
                    setSelectedIndex((current) => (current === index ? null : index))
                  }
                >
                  <span className="embedding3d-ui__swatch" />
                  <span className="embedding3d-ui__item-label">{item.label}</span>
                </button>
                <button
                  type="button"
                  className="embedding3d-ui__delete"
                  aria-label={`Delete ${item.label}`}
                  onClick={() => onDeleteItem?.(index)}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="embedding3d-ui__stage">
        <Embedding3D
          points={points}
          height={height}
          title={`${projection.modeLabel} Projection`}
        />
      </div>
    </div>
  );
}
