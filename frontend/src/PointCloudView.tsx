import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { PointCloudData } from './replay';

/** Object colors for the labeled cloud: background in neutral gray, then one hue per object. */
const LABEL_COLORS = ['#5b6068', '#ff4f00', '#3e8ef7', '#46a758', '#f5b400', '#a855f7', '#14b8a6', '#e5484d', '#ec4899'];
const FRUSTUM_EVERY = 3;

/** sRGB byte to linear light, the color space three.js works in. */
const TO_LINEAR = Float32Array.from({ length: 256 }, (_, i) => new THREE.Color().setRGB(i / 255, 0, 0, THREE.SRGBColorSpace).r);

interface PointCloudViewProps {
  data: PointCloudData;
  /** rgb: the reconstruction's own colors; label: one color per separated object. */
  colorBy: 'rgb' | 'label';
  /** Camera path, [x, y, z, qx, qy, qz, qw] per keyframe (OpenGL cameras), in the cloud's frame. */
  cameras?: number[][];
  label: string;
}

/**
 * A reconstructed point cloud with the camera path that filmed it, turning slowly. Small and
 * self-contained (three.js only), so the pipeline view doesn't pull in the simulator.
 */
export function PointCloudView({ data, colorBy, cameras, label }: PointCloudViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 50);

    const colors = new Float32Array(data.count * 3);
    const palette = LABEL_COLORS.map((hex) => new THREE.Color(hex));
    for (let i = 0; i < data.count; i++) {
      if (colorBy === 'label') {
        const c = palette[data.labels[i] === 0 ? 0 : 1 + ((data.labels[i] - 1) % (palette.length - 1))];
        colors[3 * i] = c.r;
        colors[3 * i + 1] = c.g;
        colors[3 * i + 2] = c.b;
      } else {
        colors[3 * i] = TO_LINEAR[data.colors[3 * i]];
        colors[3 * i + 1] = TO_LINEAR[data.colors[3 * i + 1]];
        colors[3 * i + 2] = TO_LINEAR[data.colors[3 * i + 2]];
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true }));
    scene.add(points);

    // Frame the middle 90% of the cloud: stray far points shouldn't shrink it to a dot.
    const center = new THREE.Vector3();
    const [xs, ys, zs]: number[][] = [[], [], []];
    for (let i = 0; i < data.count; i += Math.max(1, Math.floor(data.count / 4000))) {
      xs.push(data.positions[3 * i]);
      ys.push(data.positions[3 * i + 1]);
      zs.push(data.positions[3 * i + 2]);
    }
    const middle = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b);
      return [sorted[Math.floor(sorted.length * 0.05)], sorted[Math.floor(sorted.length * 0.95)]];
    };
    const [[x0, x1], [y0, y1], [z0, z1]] = [middle(xs), middle(ys), middle(zs)];
    center.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    const radius = Math.max(Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2, 0.05);

    if (cameras && cameras.length > 1) {
      const path = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(cameras.map(([x, y, z]) => new THREE.Vector3(x, y, z))),
        new THREE.LineBasicMaterial({ color: '#ff4f00', transparent: true, opacity: 0.9 }),
      );
      scene.add(path);
      const depth = radius * 0.12;
      const corners = [
        [-0.75, -1],
        [0.75, -1],
        [0.75, 1],
        [-0.75, 1],
      ].map(([u, v]) => new THREE.Vector3(u * depth * 0.6, v * depth * 0.45, -depth));
      const frustum: number[] = [];
      cameras.forEach(([x, y, z, qx, qy, qz, qw], k) => {
        if (k % FRUSTUM_EVERY !== 0) return;
        const origin = new THREE.Vector3(x, y, z);
        const turn = new THREE.Quaternion(qx, qy, qz, qw);
        const ends = corners.map((c) => c.clone().applyQuaternion(turn).add(origin));
        ends.forEach((end, i) => {
          frustum.push(...origin.toArray(), ...end.toArray());
          frustum.push(...end.toArray(), ...ends[(i + 1) % 4].toArray());
        });
      });
      const lines = new THREE.BufferGeometry();
      lines.setAttribute('position', new THREE.Float32BufferAttribute(frustum, 3));
      scene.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: '#ff4f00', transparent: true, opacity: 0.45 })));
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.autoRotate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    controls.autoRotateSpeed = 0.7;
    controls.enablePan = false;
    camera.position.copy(center).add(new THREE.Vector3(0.9, 0.75, 1.1).normalize().multiplyScalar(radius * 2.6));

    const resize = () => {
      const { clientWidth: width, clientHeight: height } = container;
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    // Draw only while on screen: the page may hold several of these.
    const visibility = new IntersectionObserver(([entry]) => renderer.setAnimationLoop(entry.isIntersecting ? loop : null));
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
    };
    visibility.observe(container);
    renderer.setAnimationLoop(loop);
    return () => {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      visibility.disconnect();
      controls.dispose();
      scene.traverse((node) => {
        if (node instanceof THREE.Points || node instanceof THREE.Line) {
          node.geometry.dispose();
          (node.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [data, colorBy, cameras]);

  return <div ref={containerRef} className="cloud-view" role="img" aria-label={label} />;
}
