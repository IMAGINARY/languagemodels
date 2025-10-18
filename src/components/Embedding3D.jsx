import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Html } from "@react-three/drei/web/Html";

function PointsCloud({ points }) {
  const { positions, colors } = useMemo(() => {
    if (!points?.length) return { positions: new Float32Array(), colors: new Float32Array() };
    const pos = new Float32Array(points.length * 3);
    const col = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      pos[i * 3 + 0] = p.x;
      pos[i * 3 + 1] = p.y;
      pos[i * 3 + 2] = p.z;
      const color = new THREE.Color(p.color || "#6ee7ff");
      col[i * 3 + 0] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }
    return { positions: pos, colors: col };
  }, [points]);

  return (
    <points key={positions.length} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={positions.length / 3} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={colors.length / 3} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={0.06} sizeAttenuation />
    </points>
  );
}

function Labels({ points }) {
  return (
    <group>
      {points.map((p, i) => (
        p.label ? (
          <group key={i} position={[p.x, p.y, p.z]}>
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
        ) : null
      ))}
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
          <PointsCloud points={normalizedPoints} />
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
