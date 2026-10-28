# Architecture

Darius has one data structure, the **assembly plan**, and three jobs that revolve around it: recover it from a
manual, track a build's progress through it, and turn its geometry into guidance. This document follows the plan
through each of them, with the algorithms and the numbers they use. Perception runs on two Meta models through the
Meta Model API: Muse Spark 1.3 reads manuals and frames and points at parts; SAM 3.1 tracks every part through video.

```
             ┌──────────── engine (Python) ────────────┐        ┌──────────── web (TypeScript) ─────────────┐
manual  ──►  perception.manual ──► to_plan ──► validate  ──┐    │  engine/plan.ts      (schema mirror)       │
pages        (Muse Spark, schema)                          │    │  engine/tracking.ts  (tracker mirror)      │
                                                           ▼    │  engine/choreography.ts ──► three/Stage.ts │
                                            AssemblyPlan (graph) ─►  (every instant of a build)   AR layer   │
camera  ──►  perception.frames ──► Observation ──► Tracker ─┘    │  three/ManualArt.ts ──► ui/ManualPage.tsx  │
frames       (Muse Spark + step pages)   P(step done)   HMM      │  (the plan drawn as a manual)              │
   │                                         ▲ fused             └───────────────────────────────────────────┘
   └──►  perception.parts ──► masks ──► contacts
         (Muse Spark grounds, SAM 3.1 tracks, Muse Spark re-finds)
```

## 1. The plan

`engine/darius/assembly/schema.py` defines `AssemblyPlan`:

| Field | Meaning |
|---|---|
| `parts` | one entry per physical piece: `id` (unique, `A1`), `label` (the printed letter, `A`), `name`, `kind`, `size_mm`, optional `assembled` pose and floor `layout` pose |
| `hardware` | the inventory: article number, name, kind, count, size |
| `joints` | edges: two part ids, a kind (`dowel`, `cam_lock`, `screw`, `bolt`, `nail`, `slot`, `pin`, `glue`), the hardware it uses, and its fasteners (position and insertion axis in the furniture frame) |
| `steps` | the manual's order: the parts each step brings in, the joints it makes, its hardware, actions, tools, and the `orientation` the piece lies in while you work (`upright`, `on_back`, `on_front`, `on_side`, `upside_down`) |

Units are millimeters; +y is up, the floor is y = 0 and the front faces +z. Geometry is optional: a plan read from a
manual has none, and every graph operation works without it.

`assembly/graph.py` holds the graph operations: `validate` (joints reference parts, steps bring parts in before
joining them, hardware within the inventory, the finished piece is one connected component), `subassemblies(plan, k)`
(union-find over the joints of steps 1..k), and `compare(predicted, truth)`. Scores are **by letter**, since which of
two identical side panels is `A1` is arbitrary: parts are compared as multisets of letters, joints as multisets of
sorted letter pairs, steps exactly (the multiset of a step's joint letter pairs), and hardware by count per article.

## 2. The library: ground truth from geometry

`furniture/builder.py` compiles a product declared as boxes into a plan:

1. **Contacts.** For two parts, their axis-aligned boxes in the assembled frame touch when the gap along one axis is
   within 0.6 mm and they overlap on the other two; the shared face gives the contact normal and its rectangle.
2. **Hardware patterns.** A joint's fasteners are spread along the contact rectangle's long axis: a cam-lock joint is
   dowel, cam bolt, cam bolt, dowel (each cam bolt with a cam lock); dowel joints two dowels; screws two; bolts one;
   shelf pins two per shelf end; nails one per ~150 mm (2 to 8); slots none.
3. **Floor layout.** Parts lie flat in rows in front of the work area, in the order the steps use them, so the
   first parts needed are nearest.

`catalog.py` declares four products. Their plans are exported to `assemblies/<id>/plan.json` (`darius export`) and
are the ground truth for everything measured.

## 3. Reading a manual

`perception/manual.py` sends all pages to Muse Spark in **one request** with a JSON schema (`ManualReading`): the title
and category; parts by letter with name, count and printed size; hardware by article number with kind and count; and
per step its title, the letters brought in (one entry per piece), the hardware box, the orientation drawn, the tools,
and every connection as two letters and a joint kind. Reasoning effort is `high`; a manual takes 35 to 70 s.

`to_plan(reading)` builds the graph deterministically:

- **Instances.** A letter printed with count *n* becomes `A1 … An` (or `A` alone); a step's `adds` takes the next
  unused instance of each letter.
- **Implicit base parts.** Manuals draw the part a step builds on lying there, without an arrow, so readers often
  leave it out of `adds`. A connection that needs a part no step has brought in yet brings in its next instance.
- **Resolving connections.** Each letter in a connection resolves to the instance that minimizes, in order: not
  brought in by this step, already paired with the other side, joints made in this step, joints made in total,
  then the id. This gives a part on two sides two different panels (bottom `C` joins `A1` and `A2`, not `A1` twice).
- **Hardware.** A step's hardware is shared among its joints by kind (dowels among dowel and cam-lock joints, cam bolts
  and locks among cam-lock joints, …) for display; scoring uses the step totals.
- **Validation.** `validate` runs on the result and its problems are returned as notes, never silently dropped.

## 4. Verifying frames

`perception/frames.py` sends consecutive frames (4 per request, 6 requests at a time) with:

- the parts list and, for every step, its instruction and what the build looks like once it is done (which letters
  are joined, how the piece lies), generated from the plan by `describe_plan`;
- the manual's **step pages** as reference images, first in every request.

The schema asks, per frame, for every step's `done` with a confidence, the letters joined and loose, and the
orientation. `to_observation` turns a reading into tracker evidence: P(step *i* done) = confidence if done else
1 − confidence, kept within [0.02, 0.98].

## 5. Tracking parts

`perception/parts.py` follows every physical piece through a build video with both models.

1. **Ground.** On an anchor frame (1 s in, the parts on the floor), Muse Spark sees the frame and the manual's parts
   page and returns, per piece, its letter, a box and a point on the piece (0–1000 grid). It looks once at the
   whole frame, then three times at a close crop around everything it found (a 5 px leg spans 15 px there), and each
   piece takes the letter most close looks give it. Without the point, SAM segmented the floor plank beside a thin
   diagonal leg; without the crop, Muse Spark's points missed thin legs by ~15 px.
2. **Track.** The video is re-encoded to 5 fps at 960×540 and uploaded once. One SAM 3.1 request carries every
   piece's box and point as one object's prompt on the anchor frame and propagates all masks forward and backward
   (up to 16 objects). Masks arrive box-local and are stored as simplified polygons.
3. **Find again.** SAM drops a part that leaves the frame (the session camera comes in close to fasten) and does not
   pick it up when it comes back. Every 4 s, for the parts with no mask, Muse Spark gets the anchor frame with every
   part outlined and numbered plus the later frame (set-of-mark prompting) and returns a box and point for each lost
   part it can see, off every mask SAM still holds. SAM tracks those parts once more from all their prompts, and the
   new masks only fill frames the first pass lost: SAM propagates every prompt both ways, so a wrong later prompt
   could otherwise rewrite frames the first pass had right.
4. **Contacts.** Two parts' masks that touch (within a cell at 320×180) for 1 s without a break are a held contact.
   Twins share a letter, and parts of the same size are told apart only by where they go (a bookcase's top and
   bottom are the same board), so letters whose parts agree within 10% (or 10 mm) on every sorted dimension form a
   look-alike class, and held contacts are counted by pair of classes against the joints each step adds, in the
   manual's order: P(step *i* done) = 0.2 + 0.6 · (its joints matched). That is tracker evidence independent of Muse
   Spark's readings; the two are fused per step as a product of experts (logits add, the contacts at weight 0.5).

Everything downstream of the polygons (contacts, scores) is computed from the stored file, so `--check` re-scores a
stored run exactly.

## 6. Tracking progress

The build's state is *k*, the number of completed steps (0 … N). `assembly/tracking.py` (mirrored by
`frontend/src/engine/tracking.ts`) is a hidden Markov model:

- **Transitions**, normalized per row: P(k→k) = .86, P(k→k+1) = .11, P(k→k+2) = .02, P(k→k−1) = .01.
- **Emissions.** State *k* predicts steps 1..k done and k+1..N not. Each observed p<sub>i</sub> is first capped to
  [0.15, 0.85] (one image is never conclusive), then

  &nbsp;&nbsp;&nbsp;&nbsp;log P(obs | k) = 0.7 · Σ<sub>i ≤ k</sub> log p<sub>i</sub> + 0.7 · Σ<sub>i > k</sub> log(1 − p<sub>i</sub>)

  where the 0.7 tempers evidence from the same image, whose per-step readings are correlated.
- **Live** (`Tracker`): the forward algorithm, one update per observation; the belief is P(k), its argmax the
  current step.
- **Video** (`decode`): Viterbi for the most likely path (starting at k = 0) and forward-backward, in log space, for
  the posterior at every frame. Step *k*'s completion time is the first frame where the path reaches *k*.

Given the plan, the decoded path *is* the graph's history: the joints of step *k* were made when step *k* finished.

## 7. Guidance

`frontend/src/engine/choreography.ts` turns a plan into a timeline that can be evaluated at any instant:

| Phase | Seconds | What moves |
|---|---|---|
| intro | 1.6 | nothing: the parts lie in their floor layout |
| reorient | 2.4 | when a step's orientation differs from the last one, the main assembly rotates about its center, lifted off the floor |
| locate | 2.2 | the step's parts are highlighted where they lie |
| place | 3.4 + 0.35 per extra part | each part travels on an arc (lifted mid-way, rotation slerped) from the floor to its pose; staggered |
| fasten | 2.0 | fastener markers appear one by one at each joint's hole positions |
| verify | 1.6 | the markers turn green; the step counts as done at the end |

Poses come from the plan: a part's assembled pose composed with the transform of the component it belongs to after
*k* steps (`subassemblies`). The main component lies in the step's orientation and rests on the floor at the work
area; a subassembly built on the side (the nightstand's drawer) stands upright beside it until the step that joins
it, when it travels as one piece.

`three/Stage.ts` renders a moment: the room or the studio, rounded flat-pack parts with procedural wood, and the AR
layer: ghost boxes at target poses, tube arcs whose dashes flow toward the target, a rotation ring for turns, and
screen-sized fastener sprites. It also returns **detections**: every part's projected box, with loose parts and parts
being placed boxed one by one and each (sub)assembly as a single object, which is how a part tracker reports them.

## 8. Manuals from geometry

`three/ManualArt.ts` draws a plan as a manual: an orthographic three-quarter view, white faces with polygon offset
under heavy screen-space outlines (hidden lines removed by the depth buffer), new parts white and parts in place gray.
A step's drawing shows the component the step works on, in the step's orientation, with the parts it adds pulled out:

1. **Direction.** Away from the part it joins, along the axis where their boxes touch (the axis with the largest
   gap − half-sizes).
2. **Free way in.** That direction is kept only if sweeping the part's box along it 4 m clears every part already in
   place (boxes shrunk by 1.5 mm, so touching is not hitting); otherwise the first free one of front, up, down, right,
   left, back. Shelves between two sides therefore come in from the front.
3. **Depth.** Breadth-first from the parts in place over the step's joints: a part two joints away is pulled twice as
   far. Parts that would land on or hide each other (boxes closer than 10 cm) push the outer one further.
4. **Subassemblies.** A step that joins a side subassembly pulls all of it out together, along a free way, with one
   arrow.

The drawing's callouts and arrows are returned in image coordinates and laid out as SVG over it by
`ui/ManualPage.tsx`, which also composes the cover, the parts page (all parts at one scale) and the hardware page.

## 9. Measuring

`frontend/scripts/capture.mjs` renders each product's manual pages and films its build in clean mode (no AR layer)
at 1280×720, 15 fps, with the moment of every frame recorded in `truth.json`, and renders every part's exact mask
(a label map per frame at 5 fps, 320×180, `ids.bin.gz`) by drawing each part in a flat color of its index.
`darius eval` then:

- reads each manual and scores the recovered plan with `compare`;
- samples each session at 1 fps, verifies the frames with the step pages, decodes them, and scores the timeline:
  **frame accuracy** (the decoded *k* equals the truth), the same for each frame alone (argmax of its emission),
  **steps found**, and the **timing error** of each step's decoded completion against the moment its parts were in
  place (in a rendered session nothing visible happens during fastening, so that is the moment a video can show);
- tracks each session's parts (`--tracking`) and scores the tracks against the exact masks: each track is matched to
  the part it covers on the anchor frame, then **mask IoU** (per track, over every frame where its part is visible,
  0 when the track has no mask there), **part-frames held** (IoU ≥ 0.5), **letters right** (up to look-alikes), the
  share of the plan's joints seen as held contacts, and the frame accuracy of the timeline decoded from the contacts
  alone, from the contacts fused with Muse Spark's readings, and from contacts of the exact masks (what perfect
  tracks would give).

Every output is stored next to its inputs; `--replay` re-scores stored outputs and `--check` (run by CI) fails when a
stored score no longer matches.

## 10. Serving

`engine/darius/service/api.py` (FastAPI, deployed by `engine/modal_app.py` as one CPU function that scales to zero):

| Endpoint | In | Out |
|---|---|---|
| `GET /library`, `GET /library/{id}` | | plan summaries, a plan |
| `POST /manuals` | page images (1–50) | the reading, the plan built from it, notes |
| `POST /frames` | 1–8 frames with times, a library `plan_id` (its step pages are used) or a plan as JSON (+ optional reference pages) | per-frame readings and tracker evidence |
| `POST /videos` | a video (≤ 300 MB) and a plan | the decoded timeline and every joint's date |

The tracker runs in the client, so `/frames` is stateless. Model calls can be gated with `DARIUS_ACCESS_CODE` (the
`X-Access-Code` header).
