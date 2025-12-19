import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls, Html, Line } from "@react-three/drei";

function VectorsCloud({ points }) {
  const vectors = useMemo(() => {
    const out = [];
    for (const p of points || []) {
      const start = new THREE.Vector3(
        p.x0 ?? p.startX ?? 0,
        p.y0 ?? p.startY ?? 0,
        p.z0 ?? p.startZ ?? 0
      );
      const end = new THREE.Vector3(p.x, p.y, p.z);
      const dir = end.clone().sub(start);
      const len = dir.length();
      if (len < 1e-6) continue; // skip near-zero
      const color = p.color || "#6ee7ff";
      out.push({
        start,
        end,
        dir: dir.normalize(),
        len,
        color,
        label: p.label,
        dashed: !!p.dashed,
      });
    }
    return out;
  }, [points]);

  return (
    <group>
      {vectors.map((v, i) => {
        // Arrow head dimensions relative to vector length
        const headLen = Math.max(0.04, Math.min(0.12, 0.08 * v.len)) * 0.5;
        const headRad = headLen * 0.3;
        // Cone placement so that its tip sits exactly at the end point
        const conePos = v.end.clone().addScaledVector(v.dir, -headLen * 0.5);
        const quat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          v.dir
        );
        return (
          <group key={i}>
            <Line
              points={[
                [v.start.x, v.start.y, v.start.z],
                [v.end.x, v.end.y, v.end.z],
              ]}
              color={v.color}
              lineWidth={2.5}
              dashed={v.dashed}
              dashSize={0.12}
              gapSize={0.08}
            />
            <mesh position={conePos} quaternion={quat}>
              <coneGeometry args={[headRad, headLen, 12]} />
              <meshBasicMaterial color={v.color} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

function Labels({ points }) {
  return (
    <group>
      {points.map((p, i) => {
        if (!p.label) return null;
        const sx = p.x0 ?? p.startX ?? 0;
        const sy = p.y0 ?? p.startY ?? 0;
        const sz = p.z0 ?? p.startZ ?? 0;
        const dx = (p.x ?? 0) - sx;
        const dy = (p.y ?? 0) - sy;
        const dz = (p.z ?? 0) - sz;
        const len = Math.hypot(dx, dy, dz);
        const ox = len > 0 ? dx / len : 0;
        const oy = len > 0 ? dy / len : 0;
        const oz = len > 0 ? dz / len : 0;
        const offset = 0.1; // small world-unit offset beyond arrow tip
        const px = (p.x ?? 0) + ox * offset;
        const py = (p.y ?? 0) + oy * offset;
        const pz = (p.z ?? 0) + oz * offset;
        return (
          <group key={i} position={[px, py, pz]}>
            <Html center style={{ pointerEvents: "none" }}>
              <span
                style={{
                  background: "rgba(20,24,36,0.9)",
                  border: "1px solid #262b3a",
                  color: "#cdd3e4",
                  fontSize: 12,
                  padding: "2px 6px",
                  borderRadius: 6,
                  whiteSpace: "nowrap",
                }}
              >
                {p.label}
              </span>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

export default function Embedding3D({
  points = [],
  width = 640,
  height = 360,
  title = "Embedding PCA (3D)",
}) {
  const [contextLost, setContextLost] = useState(false);
  const [glMessage, setGlMessage] = useState(null);
  const cleanupRef = useRef(null);

  const handleCreated = useCallback(({ gl }) => {
    const canvas = gl.domElement;
    const onLost = (e) => {
      e.preventDefault();
      setContextLost(true);
      setGlMessage("WebGL context lost. Attempting to restore…");
      try {
        gl.forceContextRestore?.();
      } catch (err) {
        console.warn("Failed to force context restore", err);
      }
    };
    const onRestored = () => {
      setContextLost(false);
      setGlMessage(null);
    };
    canvas.addEventListener("webglcontextlost", onLost, false);
    canvas.addEventListener("webglcontextrestored", onRestored, false);
    cleanupRef.current = () => {
      canvas.removeEventListener("webglcontextlost", onLost, false);
      canvas.removeEventListener("webglcontextrestored", onRestored, false);
    };
  }, []);

  useEffect(() => () => cleanupRef.current?.(), []);

  return (
    <div className="viz3d" style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>{title}</h3>
        <span style={{ color: "#a0a7b5", fontSize: 12 }}>
          {points.length ? `${points.length} points` : "no points"}
        </span>
      </div>
      <div style={{ width: "100%", height, position: "relative" }}>
        <Canvas
          camera={{ position: [1.8, 1.2, 1.8], fov: 50 }}
          dpr={[1, 1.6]}
          gl={{ antialias: true, powerPreference: "high-performance" }}
          onCreated={handleCreated}
        >
          <color attach="background" args={["#0b0d16"]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[3, 2, 1]} intensity={0.6} />
          <gridHelper
            args={[4, 8, "#263145", "#1c263a"]}
            position={[0, -1.1, 0]}
          />
          <axesHelper args={[1.5]} />
          <VectorsCloud points={points} />
          <Labels points={points} />
          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            rotateSpeed={0.6}
            zoomSpeed={0.8}
          />
        </Canvas>
        {glMessage ? (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(5,7,12,0.7)",
              color: "#cdd3e4",
              fontSize: 13,
              backdropFilter: "blur(1px)",
            }}
          >
            {glMessage}
          </div>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 8, color: "#a0a7b5" }}>
        <span>Drag to orbit, wheel to zoom</span>
        {contextLost ? <span>Rendering paused; resume after restore.</span> : null}
      </div>
    </div>
  );
}
