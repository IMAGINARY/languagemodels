import React, { useEffect, useMemo, useRef, useState } from "react";

export default function AttentionArcs({ tokens, attnMatrix }) {
  const containerRef = useRef(null);
  const tokenRefs = useRef([]);
  const [centers, setCenters] = useState([]);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const measure = () => {
      // Measure token centers
      const cont = containerRef.current;
      if (!cont || !tokens?.length) return;
      setWidth(cont.clientWidth);
      const cs = tokens.map((_, i) => {
        const el = tokenRefs.current[i];
        if (!el) return 0;
        return el.offsetLeft - cont.scrollLeft + el.offsetWidth / 2;
      });
      setCenters(cs);
    };
    measure(); // Initial measure
    const onResize = () => measure(); // Re-measure on resize
    const cont = containerRef.current;
    window.addEventListener("resize", onResize);
    cont?.addEventListener("scroll", measure, { passive: true });
    return () => {
      // Cleanup listeners
      window.removeEventListener("resize", onResize);
      cont?.removeEventListener("scroll", measure);
    };
  }, [tokens]);

  const MAX_ARCS_PER_TOKEN = 3;
  const arcs = useMemo(() => {
    if (!attnMatrix || !Array.isArray(attnMatrix) || attnMatrix.length === 0)
      return [];
    const seq = Math.min(attnMatrix.length, tokens?.length || 0);
    const result = [];
    for (let i = 0; i < seq; i++) {
      const row = attnMatrix[i] || [];
      const pairs = [];
      for (let j = 0; j < seq; j++) {
        if (i === j) continue;
        const v = row[j];
        if (Number.isFinite(v)) pairs.push({ j, v });
      }
      pairs.sort((a, b) => b.v - a.v);
      const take = pairs.slice(0, MAX_ARCS_PER_TOKEN);
      for (const { j, v } of take) {
        result.push({ i, j, v: Math.max(0, Math.min(1, v)) });
      }
    }
    return result;
  }, [attnMatrix, tokens]);

  const svgHeight = 300;
  const yBase = 10;

  return (
    <div style={{ marginTop: 6 }}>
      <div
        ref={containerRef}
        style={{
          position: "relative",
          whiteSpace: "nowrap",
          overflowX: "auto",
          paddingBottom: 4,
          borderBottom: "1px dashed #e0e3ea",
        }}
      >
        {tokens?.map((t, i) => (
          <code
            key={i}
            ref={(el) => (tokenRefs.current[i] = el)}
            style={{ marginRight: 6, display: "inline-block" }}
            title={`#${i}`}
          >
            {t}
          </code>
        ))}
      </div>
      <div style={{ position: "relative", height: svgHeight }}>
        <svg
          width="100%"
          height={svgHeight}
          viewBox={`0 0 ${Math.max(width, 1)} ${svgHeight}`}
          preserveAspectRatio="none"
        >
          <defs>
            <marker
              id="arrow-head"
              markerWidth="8"
              markerHeight="6"
              refX="6"
              refY="3"
              orient="auto"
            >
              <path
                d="M 0 0 L 6 3 L 0 6 z"
                style={{ fill: "context-stroke" }}
              />
            </marker>
          </defs>
          {centers.length &&
            arcs.map((a, idx) => {
              const x1 = centers[a.i] ?? 0;
              const x2 = centers[a.j] ?? 0;
              if (!Number.isFinite(x1) || !Number.isFinite(x2)) return null;
              const dx = Math.abs(x2 - x1);
              if (dx < 1) return null;
              const h = Math.max(12, dx / 2);
              const path = `M ${x1} ${yBase} C ${x1} ${yBase + h}, ${x2} ${
                yBase + h
              }, ${x2} ${yBase}`;
              const opacity = a.v;
              return (
                <path
                  key={idx}
                  d={path}
                  fill="none"
                  stroke="#04f2ffff"
                  strokeOpacity={opacity}
                  strokeWidth={1.2}
                  markerEnd="url(#arrow-head)"
                  title={`i=${a.i}, j=${a.j}, value=${opacity.toFixed(3)}`}
                />
              );
            })}
          {!attnMatrix && (
            <text x={8} y={svgHeight - 12} fill="#a0a7b5" fontSize="12">
              Compute attention to visualize arcs
            </text>
          )}
        </svg>
      </div>
    </div>
  );
}
