import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls, Html, Line } from "@react-three/drei";

function VectorsCloud({ points }) {
  const vectors = useMemo(() => {
    const out = [];
    for (const p of points || []) {
      const end = new THREE.Vector3(p.x, p.y, p.z);
      const len = end.length();
      if (len < 1e-6) continue; // skip near-zero
      const dir = end.clone().normalize();
      const color = p.color || "#6ee7ff";
      out.push({ end, dir, len, color, label: p.label });
    }
    return out;
  }, [points]);

  return (
    <group>
      {vectors.map((v, i) => {
        // Arrow head dimensions relative to vector length
        const headLen = Math.max(0.04, Math.min(0.12, 0.08 * v.len));
        const headRad = headLen * 0.4;
        // Cone placement so that its tip sits exactly at the end point
        const conePos = v.end.clone().addScaledVector(v.dir, -headLen * 0.5);
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.dir);
        return (
          <group key={i}>
            <Line
              points={[[0, 0, 0], [v.end.x, v.end.y, v.end.z]]}
              color={v.color}
              lineWidth={2.5}
              dashed={false}
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
        const len = Math.hypot(p.x, p.y, p.z);
        const ox = len > 0 ? (p.x / len) : 0;
        const oy = len > 0 ? (p.y / len) : 0;
        const oz = len > 0 ? (p.z / len) : 0;
        const offset = 0.06; // small world-unit offset beyond arrow tip
        const px = p.x + ox * offset;
        const py = p.y + oy * offset;
        const pz = p.z + oz * offset;
        return (
          <group key={i} position={[px, py, pz]}>
            <Html center style={{ pointerEvents: "none" }}>
              <span style={{
                background: "rgba(20,24,36,0.9)",
                border: "1px solid #262b3a",
                color: "#cdd3e4",
                fontSize: 12,
                padding: "2px 6px",
                borderRadius: 6,
                whiteSpace: "nowrap",
              }}>
                {p.label}
              </span>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

export default function Embedding3D({ points = [], width = 640, height = 360, title = "Embedding PCA (3D)" }) {
  const normalizedPoints = useMemo(() => {
    if (!points.length) return [];
    let maxR = 1e-6;
    for (const p of points) {
      const r = Math.hypot(p.x, p.y, p.z);
      if (r > maxR) maxR = r;
    }
    const s = maxR > 0 ? 1 / maxR : 1;
    return points.map((p) => ({ ...p, x: p.x * s, y: p.y * s, z: p.z * s }));
  }, [points]);

  return (
    <div className="viz3d" style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>{title}</h3>
        <span style={{ color: "#a0a7b5", fontSize: 12 }}>
          {points.length ? `${points.length} points` : "no points"}
        </span>
      </div>
      <div style={{ width: "100%", height }}>
        <Canvas camera={{ position: [1.8, 1.2, 1.8], fov: 50 }} gl={{ antialias: true }}>
          <color attach="background" args={["#0b0d16"]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[3, 2, 1]} intensity={0.6} />
          <gridHelper args={[4, 8, "#263145", "#1c263a"]} position={[0, -1.1, 0]} />
          <axesHelper args={[1.5]} />
          <VectorsCloud points={normalizedPoints} />
          <Labels points={normalizedPoints} />
          <OrbitControls enableDamping dampingFactor={0.08} rotateSpeed={0.6} zoomSpeed={0.8} />
        </Canvas>
      </div>
      <div style={{ display: "flex", gap: 8, color: "#a0a7b5" }}>
        <span>Drag to orbit, wheel to zoom</span>
      </div>
    </div>
  );
}
