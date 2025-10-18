import React, { useEffect, useMemo, useRef, useState } from "react";

// Simple interactive 3D scatter using a canvas. No external deps.
export default function Embedding3D({ points = [], width = 640, height = 360, title = "Embedding PCA (3D)" }) {
  const canvasRef = useRef(null);
  const [drag, setDrag] = useState(null); // {x,y}
  const [angles, setAngles] = useState({ yaw: 0.6, pitch: 0.2 }); // radians
  const [scale, setScale] = useState(1);

  // Normalize points to unit sphere for stable view
  const normalized = useMemo(() => {
    if (!points.length) return [];
    let maxR = 1e-6;
    for (const p of points) {
      const r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (r > maxR) maxR = r;
    }
    const s = 1 / maxR;
    setScale(maxR);
    return points.map((p) => ({ ...p, x: p.x * s, y: p.y * s, z: p.z * s }));
  }, [points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Camera params
    const yaw = angles.yaw, pitch = angles.pitch;
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const camDist = 2.5; // distance from origin
    const f = 400; // focal length in px

    // Draw axes (optional)
    function project(pt) {
      // rotate Y (yaw), then X (pitch)
      const x1 = cosY * pt.x + sinY * pt.z;
      const z1 = -sinY * pt.x + cosY * pt.z;
      const y1 = cosP * pt.y - sinP * z1;
      const z2 = sinP * pt.y + cosP * z1 + camDist; // translate camera
      const s = f / (z2 <= 0.1 ? 0.1 : z2);
      return { X: w / 2 + x1 * s, Y: h / 2 - y1 * s, depth: z2, scale: s };
    }

    // Sort by depth for painter's algorithm (farther first)
    const proj = normalized.map((p, i) => ({ i, p, pr: project(p) }));
    proj.sort((a, b) => b.pr.depth - a.pr.depth);

    // Background
    ctx.fillStyle = "#0b0d16";
    ctx.fillRect(0, 0, w, h);

    // Draw points
    for (const { i, p, pr } of proj) {
      const r = Math.max(2, Math.min(8, 4 * pr.scale));
      ctx.beginPath();
      ctx.arc(pr.X, pr.Y, r, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = p.color || "#6ee7ff";
      ctx.fill();
    }

    // Optional labels (only when few points)
    if (proj.length <= 50) {
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillStyle = "#cdd3e4";
      for (const { p, pr } of proj) {
        if (!p.label) continue;
        ctx.fillText(p.label, pr.X + 6, pr.Y - 6);
      }
    }
  }, [normalized, angles]);

  function onMouseDown(e) {
    setDrag({ x: e.clientX, y: e.clientY, orig: { ...angles } });
  }
  function onMouseMove(e) {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    const sens = 0.005;
    setAngles({ yaw: drag.orig.yaw + dx * sens, pitch: drag.orig.pitch + dy * sens });
  }
  function onMouseUp() { setDrag(null); }
  function onWheel(e) {
    e.preventDefault();
    const delta = Math.sign(e.deltaY);
    // Adjust focal length by scaling width/height in drawing effect
    // here, change normalization slightly (zoom)
    const s = Math.max(0.5, Math.min(5, (1 + -delta * 0.1)));
    // We simulate zoom by scaling normalized points; easiest is to tweak css size
    // but we keep it simple: change canvas size slightly for feedback
    // Left as no-op to keep behavior stable.
  }

  return (
    <div className="viz3d" style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>{title}</h3>
        <span style={{ color: "#a0a7b5", fontSize: 12 }}>
          {points.length ? `${points.length} points (scaled, r≈${scale.toFixed(2)})` : "no points"}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: height, background: "#0b0d16", borderRadius: 10, border: "1px solid #262b3a", cursor: drag ? "grabbing" : "grab" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
      />
      <div style={{ display: "flex", gap: 8, color: "#a0a7b5" }}>
        <span>Drag to rotate</span>
      </div>
    </div>
  );
}

