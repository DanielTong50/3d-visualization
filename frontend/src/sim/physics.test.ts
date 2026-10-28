import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { beforeAll, describe, expect, it } from 'vitest';
import { MATERIALS, materialChanges } from './materials';
import { initPhysicsEngine, PHYSICS_HZ, PhysicsWorld, type SimBody } from './physics';
import { BACKGROUND, checkManifest, loadScene, type LoadedScene, type SceneFiles, type SceneObject } from './scene';

// These tests run the real physics on the hand-authored scene in /scenes/sandbox, which must keep working.
const SANDBOX_DIR = fileURLToPath(new URL('../../../scenes/sandbox', import.meta.url));
// A real pipeline output (two mugs from an Objectron clip), when it is on disk.
const SCAN_DIR = fileURLToPath(new URL('../../../scenes/objectron-mugs', import.meta.url));

const diskFiles: SceneFiles = {
  text: (path) => readFile(path, 'utf8'),
  binary: async (path) => {
    const bytes = await readFile(path);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  },
};

let sandbox: LoadedScene;

beforeAll(async () => {
  await initPhysicsEngine();
  sandbox = await loadScene(SANDBOX_DIR, diskFiles);
});

function simulate(physics: PhysicsWorld, seconds: number, onStep?: () => void): void {
  for (let i = 0; i < Math.round(seconds * PHYSICS_HZ); i++) {
    physics.step();
    onStep?.();
  }
}

function find(physics: PhysicsWorld, id: string): SimBody {
  const found = physics.bodies.find((b) => b.spec.id === id);
  if (!found) throw new Error(`no object ${id}`);
  return found;
}

/** The sandbox scene with some objects removed and others changed. */
function variant(keep: Record<string, (spec: SceneObject) => SceneObject>): LoadedScene {
  const objects = sandbox.manifest.objects.filter((o) => o.id in keep).map((o) => keep[o.id](structuredClone(o)));
  return { ...sandbox, manifest: { ...sandbox.manifest, objects } };
}

function placeAt(x: number, bottomY: number, z: number, halfHeight: number) {
  return (spec: SceneObject): SceneObject => ({
    ...spec,
    pose: { position: [x, bottomY + halfHeight, z], rotation_quat: [0, 0, 0, 1] },
  });
}

describe('sandbox scene', () => {
  it('loads all five objects without warnings', () => {
    expect(sandbox.warnings).toEqual([]);
    expect(sandbox.manifest.objects.map((o) => o.label)).toEqual([
      'hardcover book',
      'tissue box',
      'coffee mug',
      'rubber ball',
      'rubber doorstop',
    ]);
    expect(sandbox.hullPoints.get('obj_05')?.length).toBe(18);
  });

  it('objects fall onto the table and rest still, without sinking or jittering', () => {
    const physics = new PhysicsWorld(sandbox);
    expect(physics.warnings).toEqual([]);
    simulate(physics, 6);
    const before = physics.bodies.map(({ body }) => body.translation());
    simulate(physics, 1);
    const restHeights: Record<string, number> = { obj_01: 0.02, obj_02: 0.045, obj_03: 0.05, obj_04: 0.035, obj_05: 0.02 };
    physics.bodies.forEach(({ spec, body }, i) => {
      const p = body.translation();
      const v = body.linvel();
      expect(Math.hypot(v.x, v.y, v.z), `${spec.label} speed`).toBeLessThan(0.005);
      expect(Math.abs(p.y - restHeights[spec.id]), `${spec.label} height`).toBeLessThan(0.002);
      const drift = Math.hypot(p.x - before[i].x, p.y - before[i].y, p.z - before[i].z);
      expect(drift, `${spec.label} drift over 1 s`).toBeLessThan(0.0005);
    });
    physics.dispose();
  });

  it('the ball bounces much higher than the box', () => {
    const physics = new PhysicsWorld(
      variant({
        obj_02: placeAt(0.25, 0.3, 0, 0.045),
        obj_04: placeAt(-0.25, 0.3, 0, 0.035),
      }),
    );
    const rebound = (id: string, halfHeight: number) => {
      const { body } = find(physics, id);
      let landed = false;
      let peak = 0;
      return () => {
        const vy = body.linvel().y;
        if (!landed && vy > 0) landed = true; // first bounce off the table
        if (landed) peak = Math.max(peak, body.translation().y - halfHeight);
        return peak;
      };
    };
    const box = rebound('obj_02', 0.045);
    const ball = rebound('obj_04', 0.035);
    let boxPeak = 0;
    let ballPeak = 0;
    simulate(physics, 0.8, () => {
      boxPeak = box();
      ballPeak = ball();
    });
    // Dropped from 30 cm: restitution 0.85 should rebound about 0.85² × 30 ≈ 22 cm; 0.1 barely lifts.
    expect(ballPeak).toBeGreaterThan(0.15);
    expect(boxPeak).toBeLessThan(0.02);
    physics.dispose();
  });

  it('a heavy object shoves a light one much more than the other way round', () => {
    const push = (moverId: string, moverX: number, targetId: string, targetX: number) => {
      const halfHeights: Record<string, number> = { obj_01: 0.02, obj_02: 0.045 };
      const physics = new PhysicsWorld(
        variant({
          [moverId]: placeAt(moverX, 0, 0, halfHeights[moverId]),
          [targetId]: placeAt(targetX, 0, 0, halfHeights[targetId]),
        }),
      );
      simulate(physics, 0.2); // settle
      const mover = find(physics, moverId).body;
      const target = find(physics, targetId).body;
      const start = target.translation().x;
      mover.setLinvel({ x: 1.5, y: 0, z: 0 }, true);
      simulate(physics, 1.5);
      const moved = Math.abs(target.translation().x - start);
      physics.dispose();
      return moved;
    };
    const lightShoved = push('obj_01', -0.35, 'obj_02', 0); // 1.2 kg book into 150 g tissue box
    const heavyShoved = push('obj_02', -0.35, 'obj_01', 0); // tissue box into book
    expect(lightShoved).toBeGreaterThan(0.05);
    expect(lightShoved).toBeGreaterThan(5 * heavyShoved);
  });

  // The reality check relies on this: an object pushed across the table stops after about
  // v0² / (2 μ g), using the manifest's friction exactly.
  it.each([
    ['book', 'obj_01', 0.02, 1.0],
    ['tissue box', 'obj_02', 0.045, 1.2],
  ])('a pushed %s slides about v0² / (2 μ g)', (_name, id, halfHeight, v0) => {
    const physics = new PhysicsWorld(variant({ [id]: placeAt(-0.3, 0, 0, halfHeight) }));
    simulate(physics, 0.3);
    const { spec, body } = find(physics, id);
    const start = body.translation().x;
    body.setLinvel({ x: v0, y: 0, z: 0 }, true);
    simulate(physics, 2);
    const slid = body.translation().x - start;
    const expected = v0 ** 2 / (2 * spec.friction * 9.81);
    expect(Math.abs(slid - expected) / expected).toBeLessThan(0.08);
    physics.dispose();
  });

  it('reset puts every object back exactly where it started', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 1);
    for (const { body } of physics.bodies) body.setLinvel({ x: 1, y: 0.5, z: -1 }, true);
    simulate(physics, 1);
    physics.reset();
    for (const { spec, body } of physics.bodies) {
      const p = body.translation();
      expect([p.x, p.y, p.z]).toEqual(spec.pose.position.map((v) => expect.closeTo(v, 6)));
    }
    physics.dispose();
  });

  it('grabbing lifts an object to the pointer, and a flick throws it', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 2);
    const mug = find(physics, 'obj_03').body;
    const top = mug.translation();
    const grabPoint = { x: top.x, y: top.y + 0.05, z: top.z }; // middle of the mug's top face
    physics.grabAt(mug, grabPoint);
    const lifted = { x: grabPoint.x, y: grabPoint.y + 0.15, z: grabPoint.z };
    physics.moveGrabTarget(lifted);
    simulate(physics, 1);
    const held = mug.translation();
    expect(Math.hypot(held.x - lifted.x, held.y + 0.05 - lifted.y, held.z - lifted.z)).toBeLessThan(0.02); // hangs from a spring: a 330 g mug sags about 8 mm

    // Flick: sweep the pointer 30 cm sideways in 0.1 s, then let go.
    for (let i = 1; i <= 12; i++) {
      physics.moveGrabTarget({ x: lifted.x + (0.3 * i) / 12, y: lifted.y, z: lifted.z });
      simulate(physics, 1 / PHYSICS_HZ);
    }
    physics.release();
    const v = mug.linvel();
    expect(v.x).toBeGreaterThan(1);
    expect(Math.hypot(v.x, v.y, v.z)).toBeLessThanOrEqual(6.0001);
    physics.dispose();
  });
});

describe('holding objects', () => {
  // Regression: pulling light objects by an edge used to make them oscillate and fly off.
  it.each([
    ['tissue box by a top corner', 'obj_02', { x: 0.12, y: 0.045, z: 0.06 }],
    ['ball by its side', 'obj_04', { x: 0.035, y: 0, z: 0 }],
    ['doorstop by its thin end', 'obj_05', { x: 0.06, y: -0.015, z: 0.02 }],
  ])('stays calm when held by the %s, and drops gently', (_name, id, corner) => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 3);
    const { body } = find(physics, id);
    const p = body.translation();
    const grabPoint = { x: p.x + corner.x, y: p.y + corner.y, z: p.z + corner.z };
    physics.grabAt(body, grabPoint);
    let fastest = 0;
    const track = () => {
      const v = body.linvel();
      fastest = Math.max(fastest, Math.hypot(v.x, v.y, v.z));
    };
    // Carry it up and sideways over half a second, then hold still for two seconds.
    for (let i = 1; i <= 60; i++) {
      physics.moveGrabTarget({ x: grabPoint.x + 0.002 * i, y: grabPoint.y + 0.001 * i, z: grabPoint.z });
      simulate(physics, 1 / PHYSICS_HZ, track);
    }
    simulate(physics, 2, track); // a held object swings a little, like a real one, then settles
    expect(fastest).toBeLessThan(1);
    const still = body.linvel();
    expect(Math.hypot(still.x, still.y, still.z)).toBeLessThan(0.05);
    physics.release();
    simulate(physics, 2);
    const landed = body.translation();
    expect(Math.hypot(landed.x - (p.x + 0.12), landed.z - p.z)).toBeLessThan(0.1); // dropped about where it was held
    physics.dispose();
  });
});

describe('carrying', () => {
  it('holds a heavy object as tightly as a light one', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 2);
    const sag = (id: string, mass: number) => {
      physics.updateObject(id, { mass_kg: mass });
      const { body } = find(physics, id);
      const p = body.translation();
      const grabPoint = { x: p.x, y: p.y, z: p.z };
      physics.grabAt(body, grabPoint, true);
      physics.moveGrabTarget({ ...grabPoint, y: grabPoint.y + 0.2 });
      simulate(physics, 1.5);
      const held = body.translation();
      physics.release();
      return grabPoint.y + 0.2 - held.y;
    };
    const light = sag('obj_02', 0.15);
    const heavy = sag('obj_01', 20); // the book, made 20 kg
    expect(light).toBeLessThan(0.02);
    expect(heavy).toBeLessThan(0.02);
    physics.dispose();
  });

  it('keeps a carried object upright, turns it on request, and throws it with the hand', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 2);
    const { body } = find(physics, 'obj_05'); // the doorstop, held by its thin end
    const p = body.translation();
    const grabPoint = { x: p.x + 0.05, y: p.y, z: p.z };
    const before = body.rotation();
    physics.grabAt(body, grabPoint, true);
    physics.moveGrabTarget({ ...grabPoint, y: grabPoint.y + 0.15 });
    simulate(physics, 1);
    const carried = body.rotation();
    expect(Math.abs(carried.x * before.x + carried.y * before.y + carried.z * before.z + carried.w * before.w)).toBeCloseTo(1, 4);
    physics.turnHeld(Math.PI / 2);
    const turned = body.rotation();
    expect(Math.abs(turned.x * before.x + turned.y * before.y + turned.z * before.z + turned.w * before.w)).toBeCloseTo(Math.SQRT1_2, 3);
    physics.release({ x: 2, y: 0.5, z: 0 });
    const v = body.linvel();
    expect(v.x).toBeCloseTo(2, 5);
    simulate(physics, 0.05);
    expect(body.angvel().y).toBeDefined(); // rotations are free again once let go
    physics.dispose();
  });
});

describe('scene variations', () => {
  it('stands a box in for a convex hull that is flat', () => {
    const flat = new Float32Array([0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0.1, 0, 0.1]); // four points in one plane
    const physics = new PhysicsWorld({ ...sandbox, hullPoints: new Map([['obj_05', flat]]) });
    const doorstop = find(physics, 'obj_05');
    expect(doorstop.shape).toBe('box-fallback');
    expect(physics.warnings).toEqual(['obj_05 (rubber doorstop): its collision shape was missing or flat, so a box stands in']);
    simulate(physics, 2);
    expect(doorstop.body.translation().y).toBeCloseTo(0.02, 2); // a 4 cm tall box resting on the table
    physics.dispose();
  });

  it('has no floor when floor_depth_m is null: things knocked off keep falling', () => {
    const scene = variant({ obj_04: placeAt(0.9, 0, 0, 0.035) }); // the ball, placed past the table edge
    const physics = new PhysicsWorld({
      ...scene,
      manifest: { ...scene.manifest, background: { ...scene.manifest.background, floor_depth_m: null } },
    });
    simulate(physics, 1);
    expect(find(physics, 'obj_04').body.translation().y).toBeLessThan(-2);
    physics.dispose();
  });
});

describe.skipIf(!existsSync(SCAN_DIR))('real scan (scenes/objectron-mugs)', () => {
  it('loads both mugs with their splats (or point clouds), and they settle on the table', async () => {
    const scan = await loadScene(SCAN_DIR, diskFiles);
    expect(scan.warnings).toEqual([]);
    const labels = scan.manifest.objects.map((o) => o.label);
    expect(labels).toHaveLength(2);
    for (const label of labels) expect(label).toMatch(/mug/);
    const visuals = [
      ...scan.manifest.objects.map((o) => [o.id, o.visual] as const),
      [BACKGROUND, scan.manifest.background.visual] as const,
    ];
    for (const [key, visual] of visuals) {
      if (visual?.type === 'splats') {
        expect(scan.splatFiles.get(key)?.bytes.byteLength, key).toBeGreaterThan(0);
      } else {
        expect(scan.pointClouds.get(key)?.getAttribute('position').count, key).toBe(visual?.count);
        expect(scan.pointClouds.get(key)?.hasAttribute('color'), key).toBe(true);
      }
    }

    const physics = new PhysicsWorld(scan);
    expect(physics.warnings).toEqual([]);
    for (const { spec, shape } of physics.bodies) expect(shape, `${spec.id} collider`).toBe('as-specified');
    simulate(physics, 3);
    const tableTop = scan.manifest.background.table_plane.point[1];
    for (const { spec, body } of physics.bodies) {
      const p = body.translation();
      const v = body.linvel();
      expect(Math.hypot(v.x, v.y, v.z), `${spec.id} speed`).toBeLessThan(0.01);
      expect(p.y, `${spec.id} height`).toBeGreaterThan(tableTop); // didn't fall through
      const start = spec.pose.position;
      expect(Math.hypot(p.x - start[0], p.z - start[2]), `${spec.id} drift`).toBeLessThan(0.05); // didn't explode
    }
    physics.dispose();
  });
});

describe('controls and "what if" actions', () => {
  it('falls slowly on the Moon and not at all with zero gravity', () => {
    const physics = new PhysicsWorld(variant({ obj_04: placeAt(0, 0.3, 0, 0.035) }));
    const ball = find(physics, 'obj_04').body;
    physics.setGravity('moon');
    simulate(physics, 0.3);
    // On Earth it would have hit the table by now; on the Moon it has dropped only ~7 cm.
    expect(ball.translation().y).toBeGreaterThan(0.25);
    physics.setGravity('zero');
    ball.setLinvel({ x: 0, y: 0, z: 0 }, true);
    const before = ball.translation().y;
    simulate(physics, 1);
    expect(Math.abs(ball.translation().y - before)).toBeLessThan(0.001);
    physics.dispose();
  });

  it('changing the material changes mass, grip and bounce', () => {
    const physics = new PhysicsWorld(sandbox);
    const mug = find(physics, 'obj_03');
    const rubber = physics.updateObject('obj_03', materialChanges(mug.spec, 'rubber'));
    expect(rubber.material).toBe('rubber');
    expect(rubber.mass_kg).toBeCloseTo(0.000554 * MATERIALS.rubber.density_kg_m3 * 0.259, 4);
    expect(mug.body.mass()).toBeCloseTo(rubber.mass_kg, 4);
    expect(mug.body.collider(0).friction()).toBeCloseTo(MATERIALS.rubber.friction, 5);
    expect(mug.body.collider(0).restitution()).toBeCloseTo(MATERIALS.rubber.restitution, 5);
    physics.dispose();
  });

  it.each([
    [40, true],
    [15, false],
  ])('tilting the table %d° makes the book slide: %s', (degrees, slides) => {
    // Friction 0.5 holds up to atan(0.5) ≈ 27° in theory. Rapier's static friction is a little
    // stickier (measured: a book at rest starts sliding between 30° and 35°), while sliding friction
    // matches theory (see the v0²/(2μg) test above), which is what the reality check measures.
    const physics = new PhysicsWorld(variant({ obj_01: placeAt(0, 0, 0, 0.02) }));
    simulate(physics, 0.3);
    const book = find(physics, 'obj_01').body;
    const start = book.translation();
    physics.tiltTable('x', degrees);
    simulate(physics, 1.5);
    expect(physics.tableTilt().degrees).toBe(degrees);
    const moved = Math.hypot(book.translation().y - start.y, book.translation().z - start.z);
    if (slides) expect(moved).toBeGreaterThan(0.05);
    else expect(moved).toBeLessThan(0.01);
    physics.reset();
    expect(physics.tableTilt().degrees).toBe(0);
    physics.dispose();
  });

  it('bumping the table jolts everything on it', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 3);
    const before = physics.bodies.map(({ body }) => body.translation());
    physics.bumpTable({ x: 1, y: 0, z: 0 }, 'medium');
    simulate(physics, 1);
    physics.bodies.forEach(({ spec, body }, i) => {
      expect(body.translation().x - before[i].x, spec.label).toBeGreaterThan(0.01);
    });
    physics.dispose();
  });

  it('dropping an object lifts it and lets it fall back', () => {
    const physics = new PhysicsWorld(sandbox);
    simulate(physics, 1);
    physics.dropObject('obj_03', 0.5);
    const mug = find(physics, 'obj_03').body;
    expect(mug.translation().y).toBeCloseTo(0.55, 3);
    simulate(physics, 2);
    expect(mug.translation().y).toBeLessThan(0.1);
    physics.dispose();
  });
});

describe('scene.json checks', () => {
  it('skips a broken object instead of rejecting the whole scene', () => {
    const raw = structuredClone(sandbox.manifest) as unknown as { objects: Record<string, unknown>[] };
    delete raw.objects[2].mass_kg;
    const { manifest, warnings } = checkManifest(raw);
    expect(manifest.objects).toHaveLength(4);
    expect(warnings).toEqual([
      'scene.json: objects[2].mass_kg should be a number from 0.001 to 200; the object was left out',
    ]);
  });

  it('keeps an object with a bad appearance, drawing its plain shape instead', () => {
    const raw = structuredClone(sandbox.manifest) as unknown as { objects: Record<string, unknown>[] };
    raw.objects[0].visual = 'obj_01.splat'; // the old string form
    const { manifest, warnings } = checkManifest(raw);
    expect(manifest.objects).toHaveLength(5);
    expect(manifest.objects[0].visual).toBeNull();
    expect(warnings).toHaveLength(1);
  });

  it('fills in schema defaults: no rotation means upright', () => {
    const raw = structuredClone(sandbox.manifest) as unknown as { objects: { pose: Record<string, unknown> }[] };
    delete raw.objects[0].pose.rotation_quat;
    expect(checkManifest(raw).manifest.objects[0].pose.rotation_quat).toEqual([0, 0, 0, 1]);
  });

  it('rejects a scene with the wrong units', () => {
    expect(() => checkManifest({ ...sandbox.manifest, units: 'feet' })).toThrow('units should be "meters"');
  });

  it('reads the filmed view, and ignores a broken one with a warning', () => {
    const view = { position: [0, 0.4, 0.3], target: [0, 0.04, 0], fov_deg: 57 };
    expect(checkManifest({ ...sandbox.manifest, view }).manifest.view).toEqual(view);
    expect(checkManifest(sandbox.manifest).manifest.view).toBeNull();
    const broken = checkManifest({ ...sandbox.manifest, view: { ...view, target: view.position } });
    expect(broken.manifest.view).toBeNull();
    expect(broken.warnings).toHaveLength(1);
  });
});

describe('scanned tables and shapes', () => {
  const withOutline = (scene: LoadedScene, polygon: [number, number][]): LoadedScene => ({
    ...scene,
    manifest: {
      ...scene.manifest,
      background: { ...scene.manifest.background, table_plane: { ...scene.manifest.background.table_plane, polygon_m: polygon } },
    },
  });

  it('uses the table outline (polygon_m) as the collider, so objects rest wherever the table is', () => {
    const scene = variant({ obj_04: placeAt(0.75, 0, 0, 0.035) }); // the ball, past size_m's 0.6 m half-width
    const outlined = new PhysicsWorld(withOutline(scene, [[-0.6, -0.375], [0.9, -0.375], [0.9, 0.375], [-0.6, 0.375]]));
    simulate(outlined, 1);
    expect(find(outlined, 'obj_04').body.translation().y).toBeGreaterThan(0.03);
    expect(outlined.tableBounds.maxX).toBe(0.9);
    outlined.dispose();
    const rectangle = new PhysicsWorld(scene);
    simulate(rectangle, 1);
    expect(find(rectangle, 'obj_04').body.translation().y).toBeLessThan(-0.1);
    rectangle.dispose();
  });

  it('holds objects up across a many-sided outline, and they slide over it as on a plain table', () => {
    // A 24-sided ellipse: the table is built from wedges that all meet at its center.
    const ellipse = Array.from({ length: 24 }, (_, i): [number, number] => {
      const angle = (2 * Math.PI * i) / 24;
      return [0.62 * Math.cos(angle), 0.4 * Math.sin(angle)];
    });
    const physics = new PhysicsWorld(withOutline(variant({ obj_01: placeAt(-0.3, 0, 0, 0.02) }), ellipse));
    simulate(physics, 0.3);
    const { spec, body } = find(physics, 'obj_01');
    const start = body.translation().x;
    const v0 = 1.6;
    body.setLinvel({ x: v0, y: 0, z: 0 }, true);
    simulate(physics, 2);
    const slid = body.translation().x - start;
    const expected = v0 ** 2 / (2 * spec.friction * 9.81);
    expect(Math.abs(slid - expected) / expected).toBeLessThan(0.08);
    expect(body.translation().y).toBeCloseTo(0.02, 2); // resting on the top, not sunk into it
    expect(Math.hypot(body.linvel().x, body.linvel().y, body.linvel().z)).toBeLessThan(0.01);
    physics.dispose();
  });

  it('builds convex_parts as one body and shares the mass out by volume', () => {
    const box = (x0: number, x1: number) => {
      const v: number[] = [];
      for (const x of [x0, x1]) for (const y of [-0.02, 0.02]) for (const z of [-0.025, 0.025]) v.push(x, y, z);
      return new Float32Array(v);
    };
    const scene = variant({
      obj_05: (spec) => ({ ...placeAt(0, 0, 0, 0.02)(spec), collider: { type: 'convex_parts', files: ['a.obj', 'b.obj'] } }),
    });
    const physics = new PhysicsWorld({ ...scene, hullParts: new Map([['obj_05', [box(-0.065, 0), box(0, 0.065)]]]) });
    const { body, shape, spec } = find(physics, 'obj_05');
    expect(shape).toBe('as-specified');
    expect(body.numColliders()).toBe(2);
    expect(body.mass()).toBeCloseTo(spec.mass_kg, 5);
    expect(body.collider(0).mass()).toBeCloseTo(spec.mass_kg / 2, 5);
    simulate(physics, 1);
    expect(body.translation().y).toBeCloseTo(0.02, 2);
    physics.dispose();
  });

  it('a poke sends a light ball much further than a heavy book', () => {
    const physics = new PhysicsWorld(variant({ obj_01: placeAt(-0.3, 0, 0, 0.02), obj_04: placeAt(0.3, 0, 0, 0.035) }));
    simulate(physics, 0.5);
    const book = find(physics, 'obj_01').body;
    const ball = find(physics, 'obj_04').body;
    const [bookStart, ballStart] = [book.translation(), ball.translation()];
    physics.poke('obj_01', book.translation(), { x: 0, y: 0, z: -1 });
    physics.poke('obj_04', ball.translation(), { x: 0, y: 0, z: -1 });
    simulate(physics, 0.3);
    const bookMoved = Math.abs(book.translation().z - bookStart.z);
    const ballMoved = Math.abs(ball.translation().z - ballStart.z);
    expect(ballMoved).toBeGreaterThan(0.3);
    expect(ballMoved).toBeGreaterThan(10 * bookMoved);
    physics.dispose();
  });
});
