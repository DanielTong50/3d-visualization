import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { parseObjVertices } from './sim/scene';

/**
 * The scene's 3D scene graph (robotics/scene_graph.json, written by darius.robotics.graph), drawn
 * where it is: each object as a node at its center of mass over its collision wireframe, its convex
 * parts and grasp candidates around it, the table that supports it, the objects near it and the
 * cameras that saw it. Drag to orbit; hover or tap a node to see what it is and what it relates to.
 */

type NodeType = 'frame' | 'surface' | 'object' | 'part' | 'grasp' | 'camera';
type Relation = 'contains' | 'supports' | 'near' | 'part_of' | 'grasp_of' | 'observes';

interface GraphNode {
  id: string;
  type: NodeType;
  position: [number, number, number];
  label?: string;
  mesh?: string | null;
  pose?: { position: [number, number, number]; rotation_quat: [number, number, number, number] };
  dimensions_m?: [number, number, number];
  [key: string]: unknown;
}

interface GraphEdge {
  source: string;
  target: string;
  relation: Relation;
  distance_m?: number;
}

interface SceneGraph {
  title: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const NODE_STYLE: Record<NodeType, { color: string; radius: number; name: string }> = {
  object: { color: '#ff4f00', radius: 0.008, name: 'objects' },
  surface: { color: '#f4f4f5', radius: 0.01, name: 'surface' },
  part: { color: '#7cc4ff', radius: 0.0035, name: 'convex parts' },
  grasp: { color: '#4cc38a', radius: 0.003, name: 'grasps' },
  camera: { color: '#a1a1aa', radius: 0.006, name: 'cameras' },
  frame: { color: '#f4f4f5', radius: 0.005, name: 'world frame' },
};

export const EDGE_STYLE: Record<Relation, { color: string; opacity: number; name: string }> = {
  supports: { color: '#f4f4f5', opacity: 0.75, name: 'supports' },
  near: { color: '#ff4f00', opacity: 0.85, name: 'near' },
  part_of: { color: '#7cc4ff', opacity: 0.4, name: 'part of' },
  grasp_of: { color: '#4cc38a', opacity: 0.35, name: 'grasp of' },
  observes: { color: '#a1a1aa', opacity: 0.12, name: 'observes' },
  contains: { color: '#f4f4f5', opacity: 0.3, name: 'contains' },
};

const WIRE_COLOR = '#3a3f47';
const DIM = 0.12;

function describe(node: GraphNode): [string, string][] {
  const rows: [string, string][] = [];
  const value = (key: string) => (typeof node[key] === 'number' ? (node[key] as number) : null);
  const [mass, friction, restitution, width, score, area] = ['mass_kg', 'friction', 'restitution', 'width_m', 'score', 'area_m2'].map(value);
  if (typeof node.material === 'string') rows.push(['material', node.material]);
  if (mass !== null) rows.push(['mass', mass < 1 ? `${Math.round(mass * 1000)} g` : `${mass.toFixed(2)} kg`]);
  if (friction !== null) rows.push(['friction', friction.toFixed(2)]);
  if (restitution !== null) rows.push(['restitution', restitution.toFixed(2)]);
  if (width !== null) rows.push(['opening', `${(width * 100).toFixed(1)} cm`]);
  if (score !== null) rows.push(['score', score.toFixed(2)]);
  if (area !== null) rows.push(['area', `${area.toFixed(3)} m²`]);
  if (node.dimensions_m) rows.push(['size', `${node.dimensions_m.map((d) => (d * 100).toFixed(1)).join(' × ')} cm`]);
  rows.push(['position', `${node.position.map((v) => (v * 100).toFixed(1)).join(', ')} cm`]);
  return rows;
}

export function SceneGraphView({ sceneDir }: { sceneDir: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [graph, setGraph] = useState<SceneGraph | null>(null);
  const [failed, setFailed] = useState(false);
  const [focus, setFocus] = useState<{ node: GraphNode; pinned: boolean } | null>(null);
  const pinned = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setGraph(null);
    setFailed(false);
    setFocus(null);
    pinned.current = null;
    fetch(`${sceneDir}/robotics/scene_graph.json`)
      .then((response) => (response.ok ? (response.json() as Promise<SceneGraph>) : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then((loaded) => !cancelled && setGraph(loaded))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [sceneDir]);

  useEffect(() => {
    const container = containerRef.current;
    if (!graph || !container) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.domElement.className = 'graph-labels';
    container.appendChild(labels.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 60);
    // The graph is in the robotics frame (+z up); three.js is +y up.
    const world = new THREE.Group();
    world.rotation.x = -Math.PI / 2;
    scene.add(world);

    const index = new Map(graph.nodes.map((node, i) => [node.id, i]));
    const at = (node: GraphNode) => new THREE.Vector3(...node.position);

    // Collision wireframes: the objects' "skeleton".
    const wireMaterial = new THREE.LineBasicMaterial({ color: WIRE_COLOR, transparent: true, opacity: 0.9 });
    const objectFrame = (node: GraphNode) => {
      const group = new THREE.Group();
      if (node.pose) {
        group.position.set(...node.pose.position);
        group.quaternion.set(...node.pose.rotation_quat);
      }
      world.add(group);
      return group;
    };
    const wireUrls: [THREE.Group, string][] = [];
    for (const node of graph.nodes) {
      if (node.type !== 'object') continue;
      const group = objectFrame(node);
      const parts = graph.edges.filter((e) => e.relation === 'part_of' && e.target === node.id).map((e) => graph.nodes[index.get(e.source)!]);
      const meshes = parts.length ? parts.map((p) => p.mesh).filter((m): m is string => !!m) : node.mesh ? [node.mesh] : [];
      if (meshes.length) {
        for (const mesh of meshes) wireUrls.push([group, mesh]);
      } else if (node.dimensions_m) {
        group.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...node.dimensions_m)), wireMaterial));
      }
    }
    let disposed = false;
    for (const [group, file] of wireUrls) {
      fetch(`${sceneDir}/robotics/${file}`)
        .then((response) => (response.ok ? response.text() : Promise.reject(new Error(file))))
        .then((text) => {
          if (disposed) return;
          const lines = text.split('\n');
          const vertices = parseObjVertices(text);
          const faces = lines
            .filter((line) => line.startsWith('f '))
            .flatMap((line) => line.split(/\s+/).slice(1, 4).map((token) => Number(token.split('/')[0]) - 1));
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
          geometry.setIndex(faces);
          group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 25), wireMaterial));
          geometry.dispose();
        })
        .catch(() => {}); // a missing wireframe leaves the node and its edges
    }

    // A metric grid on the table plane, under the objects.
    const objects = graph.nodes.filter((n) => n.type === 'object');
    const fitPoints = [...objects, ...graph.nodes.filter((n) => n.type === 'surface')].map(at);
    const box = new THREE.Box3().setFromPoints(fitPoints.length ? fitPoints : [new THREE.Vector3()]);
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.15);
    const grid = new THREE.GridHelper(radius * 4, 16, '#2a2d33', '#1a1c20');
    grid.rotation.x = Math.PI / 2; // GridHelper lies in xz; the world group's z is up
    grid.position.set(center.x, center.y, 0);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.6;
    world.add(grid);

    // Nodes: one instanced sphere per node, sized and colored by type.
    const sphere = new THREE.SphereGeometry(1, 16, 12);
    const nodes = new THREE.InstancedMesh(sphere, new THREE.MeshBasicMaterial({ transparent: true }), graph.nodes.length);
    const baseColors = graph.nodes.map((node) => new THREE.Color(NODE_STYLE[node.type].color));
    const matrix = new THREE.Matrix4();
    graph.nodes.forEach((node, i) => {
      const r = NODE_STYLE[node.type].radius;
      matrix.compose(at(node), new THREE.Quaternion(), new THREE.Vector3(r, r, r));
      nodes.setMatrixAt(i, matrix);
      nodes.setColorAt(i, baseColors[i]);
    });
    world.add(nodes);

    // Edges: straight lines, colored by relation; "near" ones bow up so they read as arcs.
    const segments: number[] = [];
    const edgeRanges: { edge: GraphEdge; start: number; count: number }[] = [];
    for (const edge of graph.edges) {
      const a = graph.nodes[index.get(edge.source) ?? -1];
      const b = graph.nodes[index.get(edge.target) ?? -1];
      if (!a || !b) continue;
      const pa = at(a);
      const pb = at(b);
      const points =
        edge.relation === 'near'
          ? new THREE.QuadraticBezierCurve3(pa, pa.clone().add(pb).multiplyScalar(0.5).add(new THREE.Vector3(0, 0, pa.distanceTo(pb) * 0.35)), pb).getPoints(16)
          : [pa, pb];
      const start = segments.length / 3;
      for (let k = 0; k < points.length - 1; k++) segments.push(...points[k].toArray(), ...points[k + 1].toArray());
      edgeRanges.push({ edge, start, count: segments.length / 3 - start });
    }
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
    edgeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array((segments.length / 3) * 4), 4));
    const edgeLines = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    world.add(edgeLines);

    // Labels for the things people name: objects and the table.
    for (const node of graph.nodes) {
      if (node.type !== 'object' && node.type !== 'surface') continue;
      const tag = document.createElement('div');
      tag.className = `graph-tag graph-tag-${node.type}`;
      tag.textContent = node.label ?? node.id;
      const label = new CSS2DObject(tag);
      label.position.copy(at(node)).add(new THREE.Vector3(0, 0, node.type === 'object' ? 0.02 : -0.012));
      world.add(label);
    }

    /** Light up `id`, its edges and its neighbors; dim the rest (null: everything at rest). */
    const highlight = (id: string | null) => {
      const neighbors = new Set<string>(id ? [id] : []);
      if (id) for (const { source, target } of graph.edges) if (source === id || target === id) neighbors.add(source).add(target);
      graph.nodes.forEach((node, i) => {
        const color = baseColors[i].clone();
        if (id && !neighbors.has(node.id)) color.multiplyScalar(DIM * 2);
        nodes.setColorAt(i, color);
      });
      if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;
      const colors = edgeGeometry.getAttribute('color') as THREE.BufferAttribute;
      for (const { edge, start, count } of edgeRanges) {
        const style = EDGE_STYLE[edge.relation];
        const c = new THREE.Color(style.color);
        const lit = !id || edge.source === id || edge.target === id;
        const alpha = id ? (lit ? Math.min(1, style.opacity + 0.35) : style.opacity * DIM) : style.opacity;
        for (let k = start; k < start + count; k++) colors.setXYZW(k, c.r, c.g, c.b, alpha);
      }
      colors.needsUpdate = true;
    };
    highlight(null);

    const controls = new OrbitControls(camera, renderer.domElement);
    const target = center.clone().applyEuler(world.rotation);
    controls.target.copy(target);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.autoRotate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    controls.autoRotateSpeed = 0.5;
    controls.addEventListener('start', () => (controls.autoRotate = false));
    camera.position.copy(target).add(new THREE.Vector3(0.75, 0.62, 1).normalize().multiplyScalar(radius * 3.1));

    const raycaster = new THREE.Raycaster();
    const pick = (event: PointerEvent): GraphNode | null => {
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(
        new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1),
        camera,
      );
      const [hit] = raycaster.intersectObject(nodes, false);
      return hit?.instanceId !== undefined ? graph.nodes[hit.instanceId] : null;
    };
    const onMove = (event: PointerEvent) => {
      const node = pick(event);
      renderer.domElement.style.cursor = node ? 'pointer' : '';
      if (pinned.current) return;
      highlight(node?.id ?? null);
      setFocus(node ? { node, pinned: false } : null);
    };
    const onClick = (event: PointerEvent) => {
      const node = pick(event);
      pinned.current = node && pinned.current !== node.id ? node.id : null;
      highlight(pinned.current);
      setFocus(pinned.current && node ? { node, pinned: true } : null);
    };
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('click', onClick as EventListener);

    const resize = () => {
      const { clientWidth: width, clientHeight: height } = container;
      if (!width || !height) return;
      renderer.setSize(width, height);
      labels.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
    };
    const visibility = new IntersectionObserver(([entry]) => renderer.setAnimationLoop(entry.isIntersecting ? loop : null));
    visibility.observe(container);

    return () => {
      disposed = true;
      renderer.setAnimationLoop(null);
      visibility.disconnect();
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('click', onClick as EventListener);
      controls.dispose();
      scene.traverse((node) => {
        if (node instanceof THREE.Mesh || node instanceof THREE.LineSegments || node instanceof THREE.InstancedMesh) {
          node.geometry.dispose();
          const material = node.material as THREE.Material | THREE.Material[];
          (Array.isArray(material) ? material : [material]).forEach((m) => m.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
      labels.domElement.remove();
    };
  }, [graph, sceneDir]);

  const legend = useMemo(() => {
    if (!graph) return null;
    const nodeCounts = new Map<NodeType, number>();
    for (const node of graph.nodes) nodeCounts.set(node.type, (nodeCounts.get(node.type) ?? 0) + 1);
    const edgeCounts = new Map<Relation, number>();
    for (const edge of graph.edges) edgeCounts.set(edge.relation, (edgeCounts.get(edge.relation) ?? 0) + 1);
    return { nodeCounts, edgeCounts };
  }, [graph]);

  const relations = focus && graph ? graph.edges.filter((e) => e.source === focus.node.id || e.target === focus.node.id) : [];
  const nameOf = (id: string) => graph?.nodes.find((n) => n.id === id)?.label ?? id;
  return (
    <div className="graph-view">
      <div ref={containerRef} className="graph-canvas" role="img" aria-label={graph ? `Scene graph of ${graph.title}` : 'Scene graph'} />
      {failed && <p className="graph-empty">This scene has no scene graph yet.</p>}
      {legend && (
        <div className="graph-legend" aria-label="Legend">
          {(Object.keys(NODE_STYLE) as NodeType[])
            .filter((type) => legend.nodeCounts.get(type))
            .map((type) => (
              <span key={type}>
                <i style={{ background: NODE_STYLE[type].color }} />
                {legend.nodeCounts.get(type)} {NODE_STYLE[type].name}
              </span>
            ))}
          <span className="graph-legend-gap" />
          {(Object.keys(EDGE_STYLE) as Relation[])
            .filter((relation) => legend.edgeCounts.get(relation) && relation !== 'contains')
            .map((relation) => (
              <span key={relation}>
                <b style={{ background: EDGE_STYLE[relation].color }} />
                {legend.edgeCounts.get(relation)} {EDGE_STYLE[relation].name}
              </span>
            ))}
        </div>
      )}
      {focus && (
        <aside className="graph-card">
          <p className="graph-card-type">
            {focus.node.type}
            {focus.pinned ? ' · pinned' : ''}
          </p>
          <h4>{focus.node.label ?? focus.node.id}</h4>
          <dl>
            {describe(focus.node).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          {relations.length > 0 && (
            <ul>
              {relations.slice(0, 8).map((e) => (
                <li key={`${e.source}-${e.relation}-${e.target}`}>
                  {e.source === focus.node.id ? '' : `${nameOf(e.source)} `}
                  <span>{EDGE_STYLE[e.relation].name}</span>
                  {e.source === focus.node.id ? ` ${nameOf(e.target)}` : ''}
                  {typeof e.distance_m === 'number' ? ` · ${(e.distance_m * 100).toFixed(1)} cm` : ''}
                </li>
              ))}
              {relations.length > 8 && <li className="faint">+ {relations.length - 8} more</li>}
            </ul>
          )}
        </aside>
      )}
    </div>
  );
}
