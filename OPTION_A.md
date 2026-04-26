# Option A — Human / background Gaussian split on NeuMan

This folder is a copy of `Deformable-3D-Gaussians/` with the minimum changes
needed to train on NeuMan sequences **with a human / background split on the
Gaussian side** (no SMPL conditioning yet — that is Option B). The
deformation MLP still takes `(x, t)` as input, but it is only applied to
Gaussians that were tagged as "human" during initialization. Background
Gaussians receive zero deltas and stay fully static.

Goal: remove wasted MLP capacity on background Gaussians and produce a
cleaner baseline to compare against Option B (SMPL-pose-conditioned
deformation).

All edits are marked in the source with `# Changed — Option A: ...` so
they are easy to grep and review.

---

## What changed, file by file

| File | What | Why |
|---|---|---|
| [utils/graphics_utils.py](utils/graphics_utils.py) | `BasicPointCloud` gains an optional `is_human: np.array = None` field. | Carry a per-point human tag from the reader into `GaussianModel`. Legacy readers that don't set it pass `None`, which preserves the original "deform everything" behaviour. |
| [scene/dataset_readers.py](scene/dataset_readers.py) | Added `_find_sparse_dir`, `_read_colmap_depth_bin`, `_read_human_mask`, `_backproject_pixels`, `_seed_human_points_from_masks`, and `readNeuManSceneInfo`. `readColmapSceneInfo` now looks in `sparse/0/` first and falls back to `sparse/` (NeuMan layout). `sceneLoadTypeCallbacks["NeuMan"]` is registered. | 1) Handle NeuMan's `sparse/` (no `/0`) layout. 2) Read NeuMan's COLMAP-MVS geometric depth (`.geometric.bin`) and binary segmentation masks. 3) Backproject masked pixels of a handful of frames into world space to produce human seed Gaussians. Returns a `BasicPointCloud` in which COLMAP points are tagged `False` and human seeds `True`. |
| [scene/__init__.py](scene/__init__.py) | Added a NeuMan detection branch that runs when either `--is_neuman` is set or when `segmentations/` + `depth_maps/` folders sit next to `sparse/`. | Route NeuMan scenes to the new reader without touching the COLMAP behaviour for non-NeuMan data. |
| [scene/gaussian_model.py](scene/gaussian_model.py) | New non-trainable `_is_human` bool buffer + `get_is_human` property. The buffer is populated from the point cloud in `create_from_pcd`, kept in lock-step with `_xyz` in `prune_points`, extended in `densification_postfix` (children inherit the parent's tag via `densify_and_clone` / `densify_and_split`), and persisted via `save_ply` / `load_ply`. | Propagate the human / background tag across the full Gaussian lifecycle. Checkpoint PLYs become self-describing. |
| [scene/deform_model.py](scene/deform_model.py) | `DeformModel.step` gains an `is_human=None` argument. When provided, only human Gaussians are passed through the MLP; background Gaussians receive zero `d_xyz`, `d_rotation`, `d_scaling`. | This is the actual "Option A" switch — the human/background split at the deformation boundary. |
| [arguments/__init__.py](arguments/__init__.py) | `ModelParams` gains `--is_neuman` (default `False`). | Explicit opt-in toggle in addition to the autodetect in `scene/__init__.py`. |
| [train.py](train.py) | `deform.step(...)` calls pass `is_human=gaussians.get_is_human` (training loop + validation loop). | Use the new routing at train time. |
| [render.py](render.py) | All seven `deform.step` / `timer.step` calls pass `is_human=gaussians.get_is_human`. | Use the new routing at inference time. |
| [train_gui.py](train_gui.py) | Both `self.deform.step(...)` calls pass `is_human=self.gaussians.get_is_human`. | Same, for the interactive GUI trainer. |

Behaviour on non-NeuMan scenes is unchanged: the `is_human` buffer stays
empty, `get_is_human` returns `None`, and `DeformModel.step(..., is_human=None)`
falls back to the original `self.deform(xyz, time_emb)` call.

---

## How the human seeding works (in plain words)

1. **Read COLMAP cameras and point cloud** the usual way (via the COLMAP reader, now with a `sparse/` fallback). Those points are tagged `is_human = False` — they almost exclusively lie on the static background, because COLMAP/SfM does not reconstruct moving subjects reliably.
2. **Pick 8 evenly-spaced training frames.** For each, load:
   - `<scene>/segmentations/<frame>.png` — binary NeuMan human mask (any non-zero pixel = human).
   - `<scene>/depth_maps/<frame>.png.geometric.bin` — COLMAP-MVS geometric depth, `float32` with an ASCII header `width&height&channels&`.
3. **Backproject** every masked pixel with valid depth: `X_cam = K⁻¹ · [u, v, 1]ᵀ · d`, then `X_world = Rᵀ_stored · (X_cam − T_stored)` (this matches the `R, T` convention used in `readColmapCameras`). Intrinsics come from `FovX/FovY` + image size, with `cx = W/2`, `cy = H/2` (exact for NeuMan's `PINHOLE` cameras).
4. **Subsample** to at most 5 000 points per frame so the seed cloud stays around 40 000 points total — enough to seed a human, small enough to not dominate densification.
5. **Merge** with the COLMAP cloud and tag the new points `is_human = True`. Store the merged cloud as `<scene>/points3D_with_human.ply` (for traceability; the `is_human` flag itself is kept in memory, not in that PLY).

All of this lives in `readNeuManSceneInfo` in [scene/dataset_readers.py](scene/dataset_readers.py).

---

## How to run

### 0. Prerequisites

- Same environment as the upstream [Deformable-3D-Gaussians README](README.md) (PyTorch + the two custom CUDA submodules — `diff-gaussian-rasterization` and `simple-knn` — installed from `submodules/`).
- A NeuMan scene on disk. The reader expects this layout (the `bike` scene already satisfies it):

  ```
  <scene>/
    images/00000.png ... 000NN.png
    sparse/cameras.txt  (or cameras.bin)
    sparse/images.txt   (or images.bin)
    sparse/points3D.ply (or .bin/.txt)
    segmentations/00000.png ... 000NN.png    ← binary human masks
    depth_maps/00000.png.geometric.bin ...   ← COLMAP MVS depth
  ```

  If you have NeuMan but the COLMAP files sit under `sparse/0/` instead of `sparse/`, either layout works.

### 1. Train

```bash
cd Deformable-3D-Gaussians-OptionA

python train.py \
    --source_path ../datasets/D-NeuMan/neuman/dataset/bike \
    --model_path  output/bike_optionA \
    --is_neuman \
    --eval
```

Notes:
- `--is_neuman` forces the NeuMan reader. You can also omit it — the autodetect in [scene/__init__.py](scene/__init__.py) triggers when `segmentations/` and `depth_maps/` exist.
- Do **not** pass `--is_blender` (its smaller `t_multires` timenet is tuned for synthetic D-NeRF clips, not real monocular video).
- Leave `--white_background` off — NeuMan frames are plain RGB.
- Consider `--is_6dof` for articulated motion; the MLP head then predicts an SE(3) twist instead of a plain translation.
- The first iterations are a warm-up where deltas are forced to zero (see `opt.warm_up = 3000` in [arguments/__init__.py](arguments/__init__.py)); expect the "Option A" routing to start having effect after that.
- At init you should see a log line like
  `[Option A] Seeded 39914 human points from 8 frames.` followed by
  `[Option A] Initialized 39914 human / 18523 background Gaussians.`

Training writes checkpoints under `output/bike_optionA/point_cloud/iteration_*/point_cloud.ply` and the deformation MLP weights under `output/bike_optionA/deform/iteration_*/deform.pth`. The `is_human` tag is saved inside the checkpoint PLY as an extra `is_human` float property.

### 2. Render / evaluate

```bash
python render.py \
    --model_path output/bike_optionA \
    --iteration -1 \
    --mode render
```

`render.py` supports the same `--mode` choices as upstream: `render` (reproject training/test cameras), `time`, `view`, `pose`, `original`, `all`. All of them pick up the `is_human` tag from the loaded PLY automatically.

For quantitative metrics:

```bash
python metrics.py --model_path output/bike_optionA
```

### 3. Sanity checks after training

- Open `output/bike_optionA/point_cloud/iteration_40000/point_cloud.ply` — the file should contain an `is_human` property.
- At low iterations, inspect the rendered images under `output/bike_optionA/train/ours_*/renders/`. Background regions should already be reasonably crisp (they are plain static 3DGS + no MLP). If they look degraded vs. the upstream run, the human/background tagging is likely mis-aligned — check that `segmentations/<frame>.png` exists for every training frame and that the masks actually cover the human.
- If `[Option A] No human seed points found` prints at init, the backprojection failed — verify the paths in `<scene>/segmentations/` and `<scene>/depth_maps/`.

### 4. Running a *non-NeuMan* dataset in this folder

All changes are backward-compatible. Training a classic D-NeRF or COLMAP scene from this fork is identical to upstream: do not pass `--is_neuman`, do not have `segmentations/` + `depth_maps/` next to `sparse/`, and the `_is_human` buffer stays empty → `DeformModel.step` takes the original code path.

---

## Known limitations of Option A

- Background is frozen by construction. Any real background motion (waving leaves, passers-by, moving camera-attached objects) will be modeled as reconstruction error rather than deformation.
- Human quality is still bounded by scalar-time conditioning. For fast or repetitive articulation, the MLP has to re-learn the pose-to-shape map from scratch at every `t`. Option B (SMPL pose conditioning) is the fix.
- Human seeds are derived from COLMAP-MVS depth, which is often noisy or missing on legs/arms. Densification will fill the gaps but takes iterations.
- Tag propagation is conservative: children inherit their parent's tag. A cloned/split background child that happens to wander onto the human surface will still be treated as background. In practice this shows up as mild under-coverage of the human, mitigated by seeding many human points up front.
