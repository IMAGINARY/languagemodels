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
  onSelectionChange,
}) {
  const [projectionMode, setProjectionMode] = useState("isometric");
  const [selectedIndexes, setSelectedIndexes] = useState([]);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const normalizedItems = useMemo(
    () =>
      items.map((item, index) => ({
        ...item,
        labelFull: item.labelFull || item.label || `#${index + 1}`,
        labelVec:
          item.labelVec || item.label || item.labelFull || `#${index + 1}`,
        vector: item.vector,
      })),
    [items]
  );

  const projection = useMemo(() => {
    if (projectionMode === "pca") return projectWithPCA(normalizedItems);
    return projectIsometrically(normalizedItems);
  }, [normalizedItems, projectionMode]);

  useEffect(() => {
    setSelectedIndexes((prev) =>
      prev.filter((index) => index >= 0 && index < normalizedItems.length)
    );
    if (hoveredIndex !== null && hoveredIndex >= normalizedItems.length) {
      setHoveredIndex(null);
    }
  }, [hoveredIndex, normalizedItems.length]);

  useEffect(() => {
    onSelectionChange?.(
      selectedIndexes.map((index) => ({
        index,
        item: normalizedItems[index],
      }))
    );
  }, [normalizedItems, onSelectionChange, selectedIndexes]);

  const points = useMemo(() => {
    return normalizedItems.map((item, index) => {
      const coords = projection.coords[index] ?? [0, 0, 0];
      const isHovered = hoveredIndex === index;
      const isSelected = selectedIndexes.includes(index);
      const hasSelection = selectedIndexes.length > 0;
      return {
        x: coords[0] ?? 0,
        y: coords[1] ?? 0,
        z: coords[2] ?? 0,
        label: item.labelVec,
        color: isHovered
          ? "#f59e0b"
          : isSelected
            ? "#f59e0b"
            : hasSelection
              ? "#4b5563"
            : "#6ee7ff",
      };
    });
  }, [hoveredIndex, normalizedItems, projection.coords, selectedIndexes]);

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

        <div className="embedding3d-ui__legend">
          <div className="embedding3d-ui__legend-item">
            <span className="embedding3d-ui__swatch" />
            <span>Single-token word</span>
          </div>
          <div className="embedding3d-ui__legend-item">
            <span className="embedding3d-ui__swatch embedding3d-ui__swatch--multi" />
            <span>Multi-token text</span>
          </div>
          <div className="embedding3d-ui__legend-item">
            <span className="embedding3d-ui__swatch embedding3d-ui__swatch--selected" />
            <span>Selection</span>
          </div>
        </div>

        <div className="embedding3d-ui__list" role="list">
          {normalizedItems.map((item, index) => {
            const isSelected = selectedIndexes.includes(index);
            const isHovered = hoveredIndex === index;
            return (
              <div
                key={`${item.labelFull}-${index}`}
                className={`embedding3d-ui__item${isSelected || isHovered ? " embedding3d-ui__item--active" : ""}`}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex((current) => (current === index ? null : current))}
              >
                <button
                  type="button"
                  className="embedding3d-ui__item-main"
                  onClick={() => {
                    setSelectedIndexes((current) =>
                      current.includes(index)
                        ? current.filter((value) => value !== index)
                        : [...current, index]
                    );
                  }}
                >
                  <span
                    className={`embedding3d-ui__swatch${item.singleToken === false ? " embedding3d-ui__swatch--multi" : ""}`}
                  />
                  <span className="embedding3d-ui__item-label">{item.labelFull}</span>
                </button>
                <button
                  type="button"
                  className="embedding3d-ui__delete"
                  aria-label={`Delete ${item.labelFull}`}
                  onClick={() => {
                    setSelectedIndexes((current) =>
                      current.filter((value) => value !== index)
                    );
                    onDeleteItem?.(index);
                  }}
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
