import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useEffect, useMemo, useRef } from "react";
import { OrbitControls as ThreeOrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

function OrbitControls() {
  const { camera, gl } = useThree();
  const controlsRef = useRef();
  useEffect(() => {
    controlsRef.current = new ThreeOrbitControls(camera, gl.domElement);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.08;
    controlsRef.current.rotateSpeed = 0.6;
    controlsRef.current.zoomSpeed = 0.8;
    return () => controlsRef.current?.dispose();
  }, [camera, gl]);
  useFrame(() => controlsRef.current?.update());
  return null;
}

function PointsCloud({ points }) {
  const geometryRef = useRef();
  const { positions, colors } = useMemo(() => {
    if (!points?.length) return { positions: new Float32Array(), colors: new Float32Array() };
    // Normalize to unit sphere for tight framing
    let maxR = 1e-6;
    for (const p of points) {
      const r = Math.hypot(p.x, p.y, p.z);
      if (r > maxR) maxR = r;
    }
    const s = maxR > 0 ? 1 / maxR : 1;
    const pos = new Float32Array(points.length * 3);
    const col = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      pos[i * 3 + 0] = p.x * s;
      pos[i * 3 + 1] = p.y * s;
      pos[i * 3 + 2] = p.z * s;
      const color = new THREE.Color(p.color || "#6ee7ff");
      col[i * 3 + 0] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }
    return { positions: pos, colors: col };
  }, [points]);

  return (
    <points>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial vertexColors size={0.06} sizeAttenuation />
    </points>
  );
}

export default function Embedding3D({ points = [], width = 640, height = 360, title = "Embedding PCA (3D)" }) {
  return (
    <div className="viz3d" style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>{title}</h3>
        <span style={{ color: "#a0a7b5", fontSize: 12 }}>
          {points.length ? `${points.length} points` : "no points"}
        </span>
      </div>
      <div style={{ width: "100%", height }}>
        <Canvas
          camera={{ position: [1.8, 1.2, 1.8], fov: 50 }}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#0b0d16"]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[3, 2, 1]} intensity={0.6} />
          <gridHelper args={[4, 8, "#263145", "#1c263a"]} position={[0, -1.1, 0]} />
          <axesHelper args={[1.5]} />
          <PointsCloud points={points} />
          <OrbitControls />
        </Canvas>
      </div>
      <div style={{ display: "flex", gap: 8, color: "#a0a7b5" }}>
        <span>Drag to orbit, wheel to zoom</span>
      </div>
    </div>
  );
}
