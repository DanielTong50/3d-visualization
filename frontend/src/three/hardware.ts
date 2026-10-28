import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Flat-pack hardware and tools as meshes (meters). A piece is built along +y with its leading end
 * (the tip that goes in first) at +y and its head at -y, so aligning +y with a fastener's insertion
 * axis seats it. Tools are built with the working tip at the origin and the body along -y.
 */

function cylinder(top: number, bottom: number, height: number, y: number, segments = 18): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(top, bottom, height, segments).translate(0, y, 0);
}

function cone(radius: number, height: number, y: number): THREE.BufferGeometry {
  return new THREE.ConeGeometry(radius, height, 14).translate(0, y, 0);
}

/** One piece's geometry, from its kind, length and radius. */
export function hardwareGeometry(kind: string, length: number, radius: number): THREE.BufferGeometry {
  const L = length;
  const r = radius;
  const parts: THREE.BufferGeometry[] = [];
  switch (kind) {
    case 'dowel': {
      // Fluted beech dowel with chamfered ends.
      parts.push(cylinder(r, r, L - 2 * r * 0.3, 0, 20));
      parts.push(cylinder(r * 0.82, r, r * 0.3, L / 2 - r * 0.15, 20), cylinder(r, r * 0.82, r * 0.3, -L / 2 + r * 0.15, 20));
      break;
    }
    case 'cam_bolt': {
      const head = r * 1.25;
      parts.push(new THREE.SphereGeometry(head, 16, 12).translate(0, -L / 2 + head, 0));
      parts.push(cylinder(r * 0.55, r * 0.55, L * 0.22, -L / 2 + head * 2 + L * 0.11)); // neck
      parts.push(cylinder(r * 0.8, r * 0.8, L * 0.5, L * 0.15)); // threaded shank
      parts.push(cone(r * 0.8, r * 1.2, L / 2 - r * 0.6));
      break;
    }
    case 'cam_lock': {
      // A zinc disc with a slot on its visible face (-y) and the cam's opening on its side.
      parts.push(cylinder(r, r, L, 0, 28));
      parts.push(new THREE.BoxGeometry(r * 1.5, L * 0.12, r * 0.22).translate(0, -L / 2 - L * 0.03, 0));
      break;
    }
    case 'screw': {
      const head = r * 2.1;
      parts.push(cylinder(r * 0.95, head, head * 0.55, -L / 2 + head * 0.275)); // countersunk head
      parts.push(cylinder(r * 0.95, r * 0.95, L * 0.78, -L / 2 + head * 0.55 + L * 0.39));
      parts.push(cone(r * 0.95, L * 0.16, L / 2 - L * 0.08));
      // Thread ridges.
      for (let i = 0; i < 7; i++) parts.push(cylinder(r * 1.12, r * 1.12, L * 0.025, -L * 0.2 + i * L * 0.09, 14));
      break;
    }
    case 'bolt': {
      const head = r * 1.75;
      parts.push(cylinder(head, head, r * 1.3, -L / 2 + r * 0.65, 6)); // hex head
      parts.push(cylinder(r, r, L - r * 1.3, r * 0.65));
      break;
    }
    case 'nail': {
      parts.push(cylinder(r * 2.6, r * 2.6, r * 0.7, -L / 2 + r * 0.35)); // flat head
      parts.push(cylinder(r, r, L * 0.86, -L / 2 + r * 0.7 + L * 0.43));
      parts.push(cone(r, L * 0.1, L / 2 - L * 0.05));
      break;
    }
    case 'shelf_pin': {
      parts.push(cylinder(r * 1.6, r * 1.6, L * 0.12, -L * 0.1, 18)); // flange
      parts.push(cylinder(r * 1.1, r * 1.1, L * 0.38, -L / 2 + L * 0.19)); // the part the shelf rests on
      parts.push(cylinder(r * 0.8, r * 0.8, L * 0.44, L / 2 - L * 0.22)); // the part in the hole
      break;
    }
    default:
      parts.push(cylinder(r, r, L, 0));
  }
  const merged = mergeGeometries(parts.map((g) => g.toNonIndexed()));
  parts.forEach((g) => g.dispose());
  merged.computeVertexNormals();
  return merged;
}

const METAL = { metalness: 0.85, roughness: 0.32 };

export function hardwareMaterial(kind: string): THREE.MeshStandardMaterial {
  switch (kind) {
    case 'dowel':
      return new THREE.MeshStandardMaterial({ color: '#d8b483', roughness: 0.75 });
    case 'screw':
      return new THREE.MeshStandardMaterial({ color: '#d8bf72', ...METAL }); // yellow zinc
    case 'bolt':
      return new THREE.MeshStandardMaterial({ color: '#3a3e45', metalness: 0.7, roughness: 0.4 }); // black oxide
    case 'shelf_pin':
      return new THREE.MeshStandardMaterial({ color: '#c8a352', ...METAL }); // brass
    default:
      return new THREE.MeshStandardMaterial({ color: '#c3c8cf', ...METAL }); // zinc
  }
}

/** The open tray the hardware waits in. */
export function trayModel(width: number, depth: number): THREE.Group {
  const group = new THREE.Group();
  const plastic = new THREE.MeshStandardMaterial({ color: '#f6f5f2', roughness: 0.6 });
  const wall = 0.004;
  const height = 0.02;
  const floor = new THREE.Mesh(new THREE.BoxGeometry(width, 0.004, depth), plastic);
  floor.position.y = 0.002;
  group.add(floor);
  for (const [w, d, x, z] of [
    [width, wall, 0, depth / 2],
    [width, wall, 0, -depth / 2],
    [wall, depth, width / 2, 0],
    [wall, depth, -width / 2, 0],
  ]) {
    const side = new THREE.Mesh(new THREE.BoxGeometry(w, height, d), plastic);
    side.position.set(x, height / 2, z);
    group.add(side);
  }
  group.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  return group;
}

/** A tool, working tip at the origin, body along -y. */
export function toolModel(tool: string): THREE.Group {
  const group = new THREE.Group();
  const steel = new THREE.MeshStandardMaterial({ color: '#c9cdd3', ...METAL });
  const yellow = new THREE.MeshStandardMaterial({ color: '#ffd200', roughness: 0.45 });
  const blue = new THREE.MeshStandardMaterial({ color: '#0058a3', roughness: 0.45 });
  const add = (geometry: THREE.BufferGeometry, material: THREE.Material) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    group.add(mesh);
    return mesh;
  };
  if (tool === 'hammer') {
    // Striking face at the origin facing +y; the handle runs along +x from the head.
    add(new THREE.BoxGeometry(0.03, 0.1, 0.03).translate(0, -0.05, 0), steel);
    add(new THREE.CylinderGeometry(0.011, 0.013, 0.26, 14).rotateZ(Math.PI / 2).translate(0.13, -0.05, 0), yellow);
    add(new THREE.CylinderGeometry(0.0135, 0.0135, 0.06, 14).rotateZ(Math.PI / 2).translate(0.23, -0.05, 0), blue);
  } else if (tool === 'allen key') {
    add(new THREE.CylinderGeometry(0.0022, 0.0022, 0.06, 6).translate(0, -0.03, 0), steel);
    add(new THREE.CylinderGeometry(0.0022, 0.0022, 0.02, 6).rotateZ(Math.PI / 2).translate(0.01, -0.06, 0), steel);
  } else {
    // Screwdriver: steel shaft, then a yellow handle with a blue band.
    add(new THREE.CylinderGeometry(0.0025, 0.0025, 0.1, 10).translate(0, -0.05, 0), steel);
    add(new THREE.CylinderGeometry(0.014, 0.012, 0.11, 16).translate(0, -0.155, 0), yellow);
    add(new THREE.CylinderGeometry(0.0145, 0.0145, 0.018, 16).translate(0, -0.13, 0), blue);
  }
  return group;
}
