import * as THREE from 'three';

/**
 * Procedural surfaces, generated on canvases at load (no image downloads): wood grain for veneers
 * and solid wood, a faint speckle for foil and hardboard, planks for the floor.
 */

function canvas(size: number, draw: (g: CanvasRenderingContext2D, size: number) => void): THREE.CanvasTexture {
  const element = document.createElement('canvas');
  element.width = element.height = size;
  const g = element.getContext('2d')!;
  draw(g, size);
  const texture = new THREE.CanvasTexture(element);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 4;
  return texture;
}

/** A deterministic pseudo-random sequence, so every load draws the same grain. */
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function shade(hex: string, amount: number): string {
  const c = new THREE.Color(hex);
  const hsl = { h: 0, s: 0, l: 0 };
  c.getHSL(hsl);
  c.setHSL(hsl.h, hsl.s, Math.min(1, Math.max(0, hsl.l + amount)));
  return `#${c.getHexString()}`;
}

function woodGrain(base: string, seed: number): THREE.CanvasTexture {
  return canvas(512, (g, size) => {
    const rand = random(seed);
    g.fillStyle = base;
    g.fillRect(0, 0, size, size);
    for (let i = 0; i < 140; i++) {
      const y = rand() * size;
      const wave = 4 + rand() * 10;
      g.strokeStyle = shade(base, (rand() - 0.6) * 0.12);
      g.globalAlpha = 0.25 + rand() * 0.35;
      g.lineWidth = 0.6 + rand() * 2.4;
      g.beginPath();
      for (let x = 0; x <= size; x += 8) g.lineTo(x, y + Math.sin((x / size) * Math.PI * 2 * (1 + rand() * 0.2) + i) * wave);
      g.stroke();
    }
    g.globalAlpha = 1;
  });
}

function speckle(base: string, seed: number, strength: number): THREE.CanvasTexture {
  return canvas(256, (g, size) => {
    const rand = random(seed);
    g.fillStyle = base;
    g.fillRect(0, 0, size, size);
    for (let i = 0; i < 2600; i++) {
      g.fillStyle = shade(base, (rand() - 0.5) * strength);
      g.fillRect(rand() * size, rand() * size, 1 + rand(), 1 + rand());
    }
  });
}

const cache = new Map<string, THREE.Material>();

/** A part's surface from its material name and color. */
export function partMaterial(material: string, color: string): THREE.Material {
  const key = `${material}|${color}`;
  if (cache.has(key)) return cache.get(key)!;
  const wood = /oak|pine|wood|birch|veneer/i.test(material);
  const map = wood
    ? woodGrain(color, color.length * 97 + material.length)
    : speckle(color, 7, material.includes('hardboard') ? 0.1 : 0.035);
  const result = new THREE.MeshStandardMaterial({
    map,
    roughness: wood ? 0.62 : material.includes('hardboard') ? 0.85 : 0.42,
    metalness: 0,
  });
  cache.set(key, result);
  return result;
}

export function floorMaterial(): THREE.MeshStandardMaterial {
  const map = canvas(1024, (g, size) => {
    const rand = random(11);
    const planks = 8;
    const width = size / planks;
    for (let i = 0; i < planks; i++) {
      for (let y = -rand() * 400; y < size; y += 380 + rand() * 260) {
        const base = shade('#b08a63', (rand() - 0.5) * 0.08);
        g.fillStyle = base;
        g.fillRect(i * width, y, width, 640);
        for (let k = 0; k < 18; k++) {
          g.strokeStyle = shade(base, (rand() - 0.6) * 0.1);
          g.globalAlpha = 0.35;
          g.lineWidth = 0.8 + rand() * 1.6;
          const x = i * width + rand() * width;
          g.beginPath();
          g.moveTo(x, y);
          g.bezierCurveTo(x + 4, y + 200, x - 4, y + 400, x + rand() * 6, y + 640);
          g.stroke();
        }
        g.globalAlpha = 1;
        g.fillStyle = 'rgba(40,25,15,0.55)';
        g.fillRect(i * width, y, width, 2);
      }
      g.fillStyle = 'rgba(40,25,15,0.6)';
      g.fillRect(i * width, 0, 2, size);
    }
  });
  map.repeat.set(4, 4);
  return new THREE.MeshStandardMaterial({ map, roughness: 0.78, metalness: 0 });
}

export function wallMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ map: speckle('#e9e6e0', 3, 0.03), roughness: 0.95 });
}
