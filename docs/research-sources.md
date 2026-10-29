# Research ledger: LEGO instructions → robot learning

Verified against primary sources on **2026-10-29**. These references inform the design; their results are not Darius benchmark results. An available paper, asset, or PDF does not automatically grant permission to redistribute it or include it in a training corpus.

## Public instruction discovery

**LEGO Customer Service — [How to download building instructions online](https://www.lego.com/en-us/service/help-topics/article/how-to-download-building-instructions-online).** LEGO describes free PDF downloads for thousands of sets, searchable by set number or keyword. This supports a set-number-first product flow. It does not support a claim that every manual is available.

**LEGO — [Building instructions](https://www.lego.com/en-us/service/building-instructions).** The official discovery surface. Preserve set number, source URL, retrieved time, booklet identifier and content hash when a document is acquired. A set can have multiple booklets and alternate builds; the first PDF is not necessarily a complete build.

**LEGO — [Fair Play](https://www.lego.com/en-us/legal/notices-and-policies/fair-play) and [Fair Play brochure](https://www.lego.com/cdn/cs/legal/assets/blt1a4c9a959ce8e1cb/LEGO_Fairplay_Nov2018.pdf).** LEGO identifies building instructions among its copyrighted materials. Darius links to the official source and keeps its distributable demonstration corpus original. Public access, model-training use and redistribution are separate questions. Record the applicable terms and permission basis before including third-party content in a released dataset.

## Geometry and reusable assets

**LDraw — [File Format Specification](https://www.ldraw.org/article/218.html).** The canonical geometry reference defines 20 LDU per stud pitch, 24 LDU per brick height and 8 LDU per plate height. It gives 0.4 mm per LDU as a real-world approximation. Native LDraw uses a right-handed coordinate system with **−Y up**. Darius uses **+Y up**; any future importer must explicitly transform axes, rotations and winding rather than reinterpret the same coordinates.

**LDraw — [Contributor Agreement](https://www.ldraw.org/article/349.html) and [Legal Info](https://www.ldraw.org/article/105).** The current contributor agreement permits CC BY 4.0 or CC0, while historical library material and other assets can carry different terms. Inspect each file's license header and the distribution's accompanying license. An LDraw part license does not establish rights to an official set's manual or to an independently authored model.

The current Darius demo uses procedural rectangular brick geometry, not a bundled copy of the LDraw library. LDraw import and arbitrary-part collision geometry remain future work.

## Manual understanding

**Wang, Zhang, Mao, Cheng and Wu — [Translating a Visual LEGO Manual to a Machine-Executable Plan](https://arxiv.org/abs/2207.12572), ECCV 2022.** MEPNet formulates manual interpretation as a sequence of component and 3D-pose predictions conditioned on the current assembly. It combines learned image keypoints with 2D–3D projection. The relevant lesson for Darius is to keep manual interpretation as a measurable module that produces a constrained geometric intermediate representation. A PDF fetch is not that module.

**Ding et al. — [JARVIS: From Visual Manuals to Autonomous Robotic Assembly](https://arxiv.org/abs/2610.27618), arXiv, submitted 2026-10-20.** The paper describes a LEGO assembly benchmark with 236,679 tasks across 8,987 scenes and a 50-task interactive suite. Its agent composes robot primitives and uses feedback for verification and recovery. Performance declines with longer assembly horizons. This is direct contemporary prior art, and motivates reporting structural correctness, order, partial progress and recovery rather than only final renders. Darius does not claim to be the first manual-to-robot system, and has not reproduced JARVIS results.

## Learning manipulation

**Zhao, Kumar, Levine and Finn — [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705), RSS 2023.** Introduces ALOHA and Action Chunking with Transformers (ACT). Predicting action chunks is a practical imitation-learning baseline for fine manipulation. Its published performance is on its own tasks and hardware, not on Darius or LEGO clutch insertion.

**Hugging Face — [ACT documentation](https://huggingface.co/docs/lerobot/main/act).** Recommends ACT as an accessible first policy and describes joint-state and multi-camera conditioning. Darius proposes ACT for the first real demonstration baseline, followed by a direct comparison with diffusion and a graph-conditioned variant. Pin the LeRobot revision used for an experiment; `main` documentation evolves.

**Chi et al. — [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), RSS 2023 / IJRR 2024.** Models a conditional distribution over action sequences and executes in a receding horizon. This is a second baseline when multiple valid grasps and approach paths matter. Darius should compare it against ACT under the same demonstrations, splits, cameras and evaluation resets before attributing a benefit to policy architecture.

**Hugging Face — [SO-101 documentation](https://huggingface.co/docs/lerobot/v0.6.1/so101).** Provides an open hardware and calibration route suitable for an initial tabletop data-collection rig. The documentation does not establish LEGO insertion precision or force-control capability. A low-cost arm should first demonstrate reliable pick-and-place in a fixture; force-sensitive insertion requires separately measured hardware capability.

**Hugging Face — [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/v0.6.1/il_robots).** Establishes the record → train → evaluate workflow and robot-specific calibration. Darius' proposed physical experiment follows this sequence, with a separate calibrated robot adapter. No browser animation is sent directly to a physical arm.

**Hugging Face — [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/v0.6.1/lerobot-dataset-v3).** The dataset contract uses tabular sensorimotor data, encoded camera video, and metadata describing features, timing and episode boundaries. A Darius JSONL trajectory is a project interchange format. It needs an explicit adapter, camera frames and hardware-specific state/action definitions before it becomes a LeRobot training dataset.

## Simulation

**MuJoCo — [Modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).** MJCF describes bodies, joints, geometry, contacts and actuators. Exporting loadable XML is useful for scene validation, but does not validate a clutch model, a manipulator controller, inverse kinematics, or transfer to a physical robot. Darius' primitive scene is a starting point for that engineering work.

**NVIDIA — [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/v2.0.0/index.html).** Describes parallel simulation, rendering and domain randomization. Isaac Lab is a possible scale-up backend after a single insertion task and its contact model have been validated. It is not a dependency of the local Darius demo.

## Research questions Darius can answer

1. Does exact stud-grid validation reduce invalid manual predictions compared with free-form pose generation?
2. Does conditioning a placement policy on a local connector graph improve generalization to unseen assemblies of known parts?
3. Can an independent post-placement verifier reduce silent failures over long build horizons?
4. How much real correction data is needed after synthetic pretraining?
5. Which failures come from perception, planning, grasping, insertion or verification?

Answer these with held-out builds, recorded resets and a disclosed hardware setup. More generated episodes alone do not answer them.
