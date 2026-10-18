# Run logs

Four parts, and they must not be mixed.

- **Part 1 — VOID.** The `curves-v1`/`curves-v2` runs, kept for provenance only.
  Every faithfulness, gate, ladder and matched-control figure in it is wrong.
  Cite nothing from it.
- **Part 2 — the first corrected run** under `curves-v3`. Superseded by Part 3,
  which reproduces it bit-for-bit, so it is kept rather than retracted. Read it
  for the history of the correction, not for figures.
- **Part 3 — seed 0, commit `34e351a`.** The full single-seed result, with the
  geometry-matched gate. This is where each quantity is defined, where the
  gate's failure is worked out in detail, and where the per-pair degradation
  table lives. Read it for what the instrument measures.
- **Part 4 — seeds 1 and 2, commit `bf477ec`.** The pre-registered three-seed
  replication (P3b.4), reported as a cross-seed summary over all three seeds.
  Read it for what holds across seeds and what does not.

**Parts 3 and 4 together are what to cite.** Part 3 is the single-seed detail;
Part 4 is the replication and the only place where cross-seed stability, and
H1a's unresolved primary-scheme test, are stated.

---

## Part 1 — VOID, kept for provenance

**Every faithfulness, gate, ladder and matched-control number in Part 1 is wrong.**
These runs were produced with `curves-v1`/`curves-v2`, whose
`_get_removal_mask` selected the pixels of *lowest* attribution magnitude
rather than the highest, so each removal curve is the curve for deleting the
background. Measured: 0 of 52 selected pixels were in the top 52 by magnitude,
all 52 were in the bottom 52. See the `curves-v3` note in
`shiftprofile/curves.py` and amendment A7 in `prereg/preregistration.md`.

Do not cite any figure from this file. It is kept because the shape of the
error is instructive — the ordering of the arms tracks where each method's
*minimum* sits, and the random-attribution control reads ≈0 throughout,
because for an i.i.d. map the bottom *k* pixels are as random as the top *k*.
That is what a correct-looking control over an inverted instrument looks like.

The calibration block is unaffected: it is computed from predictions and reads
no curve.

---

Cloning into '/kaggle/working/shiftprofile'...
remote: Enumerating objects: 85, done.
remote: Counting objects: 100% (85/85), done.
remote: Compressing objects: 100% (82/82), done.
remote: Total 85 (delta 1), reused 55 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (85/85), 183.03 KiB | 2.32 MiB/s, done.
Resolving deltas: 100% (1/1), done.
repo        /kaggle/working/shiftprofile
cache read  (none: /kaggle/input/shiftprofile-cache is not attached)
config      /kaggle/working/shiftprofile/configs/pilot.yaml
device      cuda
budget      600 min
cache write /kaggle/working/cache
cache read  (none)
data        /kaggle/working/data
corrupt     /kaggle/input/datasets/yoadjei/cifar-10-c/CIFAR-10-C
stages      predict, explain, curves
manifest    /kaggle/working/cache/run_manifest.jsonl


============================================================
Completed 0 cells, skipped 70 cached, failed 0 cells
Elapsed time: 0.2s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/configs/pilot.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean
explainers  integrated_gradients, grad_cam, random

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8610    0.2166    0.0516    0.0764    0.0315
   1     0.8330    0.2522    0.0619    0.0790    0.0430
   3     0.7247    0.4160    0.1318    0.1495    0.1114
   5     0.5487    0.6810    0.2514    0.2650    0.2702

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================
  grad_cam               mean -0.29145   0/10 cells with interval above 0  NO CELL BEATS RANDOM
  integrated_gradients   mean -0.10945   0/10 cells with interval above 0  NO CELL BEATS RANDOM
  random                 mean -0.00006   0/10 cells with interval above 0  NO CELL BEATS RANDOM

explainer               sev   faithfulness   half_width
grad_cam                  0       -0.35192      0.01655
grad_cam                  1       -0.33593      0.01708
grad_cam                  3       -0.27870      0.01849
grad_cam                  5       -0.23956      0.01785
integrated_gradients      0       -0.13794      0.00934
integrated_gradients      1       -0.12986      0.00915
integrated_gradients      3       -0.10688      0.00889
integrated_gradients      5       -0.08209      0.00822
random                    0       -0.00009      0.00012
random                    1       -0.00006      0.00014
random                    3       -0.00007      0.00013
random                    5       -0.00004      0.00012

========================================================================
P3 GATE   half-width < 10% of the clean-to-severity-5 change
========================================================================
explainer              family                 change     half_w     ratio  verdict
grad_cam               defocus_blur         +0.03556    0.01655     0.466  FAIL
grad_cam               fog                  +0.04796    0.01655     0.345  FAIL
grad_cam               gaussian_noise       +0.25354    0.02146     0.085  PASS
integrated_gradients   defocus_blur         +0.05013    0.00934     0.186  FAIL
integrated_gradients   fog                  +0.03502    0.00934     0.267  FAIL
integrated_gradients   gaussian_noise       +0.08239    0.00934     0.113  FAIL
random                 defocus_blur         -0.00000    0.00012    67.898  FAIL
random                 fog                  +0.00001    0.00012    10.066  FAIL
random                 gaussian_noise       +0.00013    0.00014     1.096  FAIL

GATE: 1/9 explainer-family pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================
add Codeadd Markdown

add Codeadd Markdown
!rm -rf /kaggle/working/shiftprofile && git clone --depth 1 https://github.com/yoadjei/shiftprofile.git /kaggle/working/shiftprofile && python /kaggle/working/shiftprofile/scripts/run_fill.py --config /kaggle/working/shiftprofile/configs/pilot_e6_imputations.yaml ; python /kaggle/working/shiftprofile/scripts/pilot_report.py --config /kaggle/working/shiftprofile/configs/pilot_e6_imputations.yaml

Cloning into '/kaggle/working/shiftprofile'...
remote: Enumerating objects: 86, done.
remote: Counting objects: 100% (86/86), done.
remote: Compressing objects: 100% (83/83), done.
remote: Total 86 (delta 1), reused 53 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (86/86), 185.42 KiB | 2.29 MiB/s, done.
Resolving deltas: 100% (1/1), done.
repo        /kaggle/working/shiftprofile
cache read  (none: /kaggle/input/shiftprofile-cache is not attached)
config      /kaggle/working/shiftprofile/configs/pilot_e6_imputations.yaml
device      cuda
budget      600 min
cache write /kaggle/working/cache
cache read  (none)
data        /kaggle/working/data
corrupt     /kaggle/input/datasets/yoadjei/cifar-10-c/CIFAR-10-C
stages      predict, explain, curves
manifest    /kaggle/working/cache/run_manifest.jsonl

  predict: predict:resnet18:0:clean:0
  model resnet18 seed 0: loading from cache or training
 61%|████████████████████████▌               | 105M/170M [20:11<16:03, 68.5kB/s]
add Codeadd Markdown
!rm -rf /kaggle/working/shiftprofile && git clone --depth 1 -b p3b-validity-pivot https://github.com/yoadjei/shiftprofile.git /kaggle/working/shiftprofile && python /kaggle/working/shiftprofile/scripts/run_fill.py --config /kaggle/working/shiftprofile/configs/pilot_validity.yaml ; python /kaggle/working/shiftprofile/scripts/pilot_report.py --config /kaggle/working/shiftprofile/configs/pilot_validity.yaml

Cloning into '/kaggle/working/shiftprofile'...
remote: Enumerating objects: 87, done.
remote: Counting objects: 100% (87/87), done.
remote: Compressing objects: 100% (84/84), done.
remote: Total 87 (delta 1), reused 48 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (87/87), 211.04 KiB | 4.80 MiB/s, done.
Resolving deltas: 100% (1/1), done.
repo        /kaggle/working/shiftprofile
cache read  (none: /kaggle/input/shiftprofile-cache is not attached)
config      /kaggle/working/shiftprofile/configs/pilot_validity.yaml
device      cuda
budget      600 min
cache write /kaggle/working/cache
cache read  (none)
data        /kaggle/working/data
corrupt     /kaggle/input/datasets/yoadjei/cifar-10-c/CIFAR-10-C
stages      predict, explain, curves
manifest    /kaggle/working/cache/run_manifest.jsonl

  predict: predict:resnet18:0:clean:0
  model resnet18 seed 0: loading from cache or training
100%|█████████████████████████████████████████| 170M/170M [14:50<00:00, 192kB/s]
    epoch 1/50, loss 1.5867
    epoch 5/50, loss 0.4779
    epoch 10/50, loss 0.1667
    epoch 15/50, loss 0.0869
    epoch 20/50, loss 0.0718
    epoch 25/50, loss 0.0517
    epoch 30/50, loss 0.0366
    epoch 35/50, loss 0.0062
    epoch 40/50, loss 0.0013
    epoch 45/50, loss 0.0015
    epoch 50/50, loss 0.0016
  predict: predict:resnet18:0:gaussian_noise:1
  predict: predict:resnet18:0:gaussian_noise:3
  predict: predict:resnet18:0:gaussian_noise:5
  predict: predict:resnet18:0:defocus_blur:1
  predict: predict:resnet18:0:defocus_blur:3
  predict: predict:resnet18:0:defocus_blur:5
  predict: predict:resnet18:0:fog:1
  predict: predict:resnet18:0:fog:3
  predict: predict:resnet18:0:fog:5
  explain: explain:resnet18:0:clean:0:integrated_gradients
  explain: explain:resnet18:0:clean:0:grad_cam
  explain: explain:resnet18:0:clean:0:random
  explain: explain:resnet18:0:clean:0:integrated_gradients_rolled
  explain: explain:resnet18:0:clean:0:grad_cam_rolled
  explain: explain:resnet18:0:clean:0:random_lowres_2
  explain: explain:resnet18:0:clean:0:random_lowres_4
  explain: explain:resnet18:0:clean:0:random_lowres_8
  explain: explain:resnet18:0:clean:0:random_lowres_16
  explain: explain:resnet18:0:clean:0:random_lowres_32
  explain: explain:resnet18:0:gaussian_noise:1:integrated_gradients
  explain: explain:resnet18:0:gaussian_noise:1:grad_cam
  explain: explain:resnet18:0:gaussian_noise:1:random
  explain: explain:resnet18:0:gaussian_noise:1:integrated_gradients_rolled
  explain: explain:resnet18:0:gaussian_noise:1:grad_cam_rolled
  explain: explain:resnet18:0:gaussian_noise:1:random_lowres_2
  explain: explain:resnet18:0:gaussian_noise:1:random_lowres_4
  explain: explain:resnet18:0:gaussian_noise:1:random_lowres_8
  explain: explain:resnet18:0:gaussian_noise:1:random_lowres_16
  explain: explain:resnet18:0:gaussian_noise:1:random_lowres_32
  explain: explain:resnet18:0:gaussian_noise:3:integrated_gradients
  explain: explain:resnet18:0:gaussian_noise:3:grad_cam
  explain: explain:resnet18:0:gaussian_noise:3:random
  explain: explain:resnet18:0:gaussian_noise:3:integrated_gradients_rolled
  explain: explain:resnet18:0:gaussian_noise:3:grad_cam_rolled
  explain: explain:resnet18:0:gaussian_noise:3:random_lowres_2
  explain: explain:resnet18:0:gaussian_noise:3:random_lowres_4
  explain: explain:resnet18:0:gaussian_noise:3:random_lowres_8
  explain: explain:resnet18:0:gaussian_noise:3:random_lowres_16
  explain: explain:resnet18:0:gaussian_noise:3:random_lowres_32
  explain: explain:resnet18:0:gaussian_noise:5:integrated_gradients
  explain: explain:resnet18:0:gaussian_noise:5:grad_cam
  explain: explain:resnet18:0:gaussian_noise:5:random
  explain: explain:resnet18:0:gaussian_noise:5:integrated_gradients_rolled
  explain: explain:resnet18:0:gaussian_noise:5:grad_cam_rolled
  explain: explain:resnet18:0:gaussian_noise:5:random_lowres_2
  explain: explain:resnet18:0:gaussian_noise:5:random_lowres_4
  explain: explain:resnet18:0:gaussian_noise:5:random_lowres_8
  explain: explain:resnet18:0:gaussian_noise:5:random_lowres_16
  explain: explain:resnet18:0:gaussian_noise:5:random_lowres_32
  explain: explain:resnet18:0:defocus_blur:1:integrated_gradients
  explain: explain:resnet18:0:defocus_blur:1:grad_cam
  explain: explain:resnet18:0:defocus_blur:1:random
  explain: explain:resnet18:0:defocus_blur:1:integrated_gradients_rolled
  explain: explain:resnet18:0:defocus_blur:1:grad_cam_rolled
  explain: explain:resnet18:0:defocus_blur:1:random_lowres_2
  explain: explain:resnet18:0:defocus_blur:1:random_lowres_4
  explain: explain:resnet18:0:defocus_blur:1:random_lowres_8
  explain: explain:resnet18:0:defocus_blur:1:random_lowres_16
  explain: explain:resnet18:0:defocus_blur:1:random_lowres_32
  explain: explain:resnet18:0:defocus_blur:3:integrated_gradients
  explain: explain:resnet18:0:defocus_blur:3:grad_cam
  explain: explain:resnet18:0:defocus_blur:3:random
  explain: explain:resnet18:0:defocus_blur:3:integrated_gradients_rolled
  explain: explain:resnet18:0:defocus_blur:3:grad_cam_rolled
  explain: explain:resnet18:0:defocus_blur:3:random_lowres_2
  explain: explain:resnet18:0:defocus_blur:3:random_lowres_4
  explain: explain:resnet18:0:defocus_blur:3:random_lowres_8
  explain: explain:resnet18:0:defocus_blur:3:random_lowres_16
  explain: explain:resnet18:0:defocus_blur:3:random_lowres_32
  explain: explain:resnet18:0:defocus_blur:5:integrated_gradients
  explain: explain:resnet18:0:defocus_blur:5:grad_cam
  explain: explain:resnet18:0:defocus_blur:5:random
  explain: explain:resnet18:0:defocus_blur:5:integrated_gradients_rolled
  explain: explain:resnet18:0:defocus_blur:5:grad_cam_rolled
  explain: explain:resnet18:0:defocus_blur:5:random_lowres_2
  explain: explain:resnet18:0:defocus_blur:5:random_lowres_4
  explain: explain:resnet18:0:defocus_blur:5:random_lowres_8
  explain: explain:resnet18:0:defocus_blur:5:random_lowres_16
  explain: explain:resnet18:0:defocus_blur:5:random_lowres_32
  explain: explain:resnet18:0:fog:1:integrated_gradients
  explain: explain:resnet18:0:fog:1:grad_cam
  explain: explain:resnet18:0:fog:1:random
  explain: explain:resnet18:0:fog:1:integrated_gradients_rolled
  explain: explain:resnet18:0:fog:1:grad_cam_rolled
  explain: explain:resnet18:0:fog:1:random_lowres_2
  explain: explain:resnet18:0:fog:1:random_lowres_4
  explain: explain:resnet18:0:fog:1:random_lowres_8
  explain: explain:resnet18:0:fog:1:random_lowres_16
  explain: explain:resnet18:0:fog:1:random_lowres_32
  explain: explain:resnet18:0:fog:3:integrated_gradients
  explain: explain:resnet18:0:fog:3:grad_cam
  explain: explain:resnet18:0:fog:3:random
  explain: explain:resnet18:0:fog:3:integrated_gradients_rolled
  explain: explain:resnet18:0:fog:3:grad_cam_rolled
  explain: explain:resnet18:0:fog:3:random_lowres_2
  explain: explain:resnet18:0:fog:3:random_lowres_4
  explain: explain:resnet18:0:fog:3:random_lowres_8
  explain: explain:resnet18:0:fog:3:random_lowres_16
  explain: explain:resnet18:0:fog:3:random_lowres_32
  explain: explain:resnet18:0:fog:5:integrated_gradients
  explain: explain:resnet18:0:fog:5:grad_cam
  explain: explain:resnet18:0:fog:5:random
  explain: explain:resnet18:0:fog:5:integrated_gradients_rolled
  explain: explain:resnet18:0:fog:5:grad_cam_rolled
  explain: explain:resnet18:0:fog:5:random_lowres_2
  explain: explain:resnet18:0:fog:5:random_lowres_4
  explain: explain:resnet18:0:fog:5:random_lowres_8
  explain: explain:resnet18:0:fog:5:random_lowres_16
  explain: explain:resnet18:0:fog:5:random_lowres_32
  curves: curves:resnet18:0:clean:0:integrated_gradients:mean
  curves: curves:resnet18:0:clean:0:integrated_gradients:blur
  curves: curves:resnet18:0:clean:0:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:clean:0:integrated_gradients:black
  curves: curves:resnet18:0:clean:0:grad_cam:mean
  curves: curves:resnet18:0:clean:0:grad_cam:blur
  curves: curves:resnet18:0:clean:0:grad_cam:uniform_noise
  curves: curves:resnet18:0:clean:0:grad_cam:black
  curves: curves:resnet18:0:clean:0:random:mean
  curves: curves:resnet18:0:clean:0:random:blur
  curves: curves:resnet18:0:clean:0:random:uniform_noise
  curves: curves:resnet18:0:clean:0:random:black
  curves: curves:resnet18:0:clean:0:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:clean:0:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:clean:0:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:clean:0:integrated_gradients_rolled:black
  curves: curves:resnet18:0:clean:0:grad_cam_rolled:mean
  curves: curves:resnet18:0:clean:0:grad_cam_rolled:blur
  curves: curves:resnet18:0:clean:0:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:clean:0:grad_cam_rolled:black
  curves: curves:resnet18:0:clean:0:random_lowres_2:mean
  curves: curves:resnet18:0:clean:0:random_lowres_2:blur
  curves: curves:resnet18:0:clean:0:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:clean:0:random_lowres_2:black
  curves: curves:resnet18:0:clean:0:random_lowres_4:mean
  curves: curves:resnet18:0:clean:0:random_lowres_4:blur
  curves: curves:resnet18:0:clean:0:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:clean:0:random_lowres_4:black
  curves: curves:resnet18:0:clean:0:random_lowres_8:mean
  curves: curves:resnet18:0:clean:0:random_lowres_8:blur
  curves: curves:resnet18:0:clean:0:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:clean:0:random_lowres_8:black
  curves: curves:resnet18:0:clean:0:random_lowres_16:mean
  curves: curves:resnet18:0:clean:0:random_lowres_16:blur
  curves: curves:resnet18:0:clean:0:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:clean:0:random_lowres_16:black
  curves: curves:resnet18:0:clean:0:random_lowres_32:mean
  curves: curves:resnet18:0:clean:0:random_lowres_32:blur
  curves: curves:resnet18:0:clean:0:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:clean:0:random_lowres_32:black
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients:mean
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients:blur
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients:black
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam:mean
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam:blur
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam:black
  curves: curves:resnet18:0:gaussian_noise:1:random:mean
  curves: curves:resnet18:0:gaussian_noise:1:random:blur
  curves: curves:resnet18:0:gaussian_noise:1:random:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random:black
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:integrated_gradients_rolled:black
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:grad_cam_rolled:black
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_2:mean
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_2:blur
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_2:black
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_4:mean
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_4:blur
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_4:black
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_8:mean
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_8:blur
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_8:black
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_16:mean
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_16:blur
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_16:black
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_32:mean
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_32:blur
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:1:random_lowres_32:black
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients:mean
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients:blur
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients:black
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam:mean
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam:blur
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam:black
  curves: curves:resnet18:0:gaussian_noise:3:random:mean
  curves: curves:resnet18:0:gaussian_noise:3:random:blur
  curves: curves:resnet18:0:gaussian_noise:3:random:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random:black
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:integrated_gradients_rolled:black
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:grad_cam_rolled:black
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_2:mean
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_2:blur
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_2:black
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_4:mean
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_4:blur
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_4:black
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_8:mean
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_8:blur
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_8:black
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_16:mean
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_16:blur
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_16:black
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_32:mean
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_32:blur
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:3:random_lowres_32:black
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients:mean
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients:blur
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients:black
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam:mean
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam:blur
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam:black
  curves: curves:resnet18:0:gaussian_noise:5:random:mean
  curves: curves:resnet18:0:gaussian_noise:5:random:blur
  curves: curves:resnet18:0:gaussian_noise:5:random:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random:black
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:integrated_gradients_rolled:black
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam_rolled:mean
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam_rolled:blur
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:grad_cam_rolled:black
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_2:mean
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_2:blur
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_2:black
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_4:mean
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_4:blur
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_4:black
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_8:mean
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_8:blur
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_8:black
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_16:mean
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_16:blur
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_16:black
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_32:mean
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_32:blur
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:gaussian_noise:5:random_lowres_32:black
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients:mean
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients:blur
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients:black
  curves: curves:resnet18:0:defocus_blur:1:grad_cam:mean
  curves: curves:resnet18:0:defocus_blur:1:grad_cam:blur
  curves: curves:resnet18:0:defocus_blur:1:grad_cam:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:grad_cam:black
  curves: curves:resnet18:0:defocus_blur:1:random:mean
  curves: curves:resnet18:0:defocus_blur:1:random:blur
  curves: curves:resnet18:0:defocus_blur:1:random:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random:black
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:integrated_gradients_rolled:black
  curves: curves:resnet18:0:defocus_blur:1:grad_cam_rolled:mean
  curves: curves:resnet18:0:defocus_blur:1:grad_cam_rolled:blur
  curves: curves:resnet18:0:defocus_blur:1:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:grad_cam_rolled:black
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_2:mean
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_2:blur
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_2:black
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_4:mean
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_4:blur
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_4:black
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_8:mean
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_8:blur
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_8:black
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_16:mean
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_16:blur
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_16:black
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_32:mean
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_32:blur
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:defocus_blur:1:random_lowres_32:black
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients:mean
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients:blur
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients:black
  curves: curves:resnet18:0:defocus_blur:3:grad_cam:mean
  curves: curves:resnet18:0:defocus_blur:3:grad_cam:blur
  curves: curves:resnet18:0:defocus_blur:3:grad_cam:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:grad_cam:black
  curves: curves:resnet18:0:defocus_blur:3:random:mean
  curves: curves:resnet18:0:defocus_blur:3:random:blur
  curves: curves:resnet18:0:defocus_blur:3:random:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random:black
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:integrated_gradients_rolled:black
  curves: curves:resnet18:0:defocus_blur:3:grad_cam_rolled:mean
  curves: curves:resnet18:0:defocus_blur:3:grad_cam_rolled:blur
  curves: curves:resnet18:0:defocus_blur:3:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:grad_cam_rolled:black
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_2:mean
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_2:blur
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_2:black
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_4:mean
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_4:blur
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_4:black
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_8:mean
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_8:blur
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_8:black
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_16:mean
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_16:blur
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_16:black
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_32:mean
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_32:blur
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:defocus_blur:3:random_lowres_32:black
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients:mean
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients:blur
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients:black
  curves: curves:resnet18:0:defocus_blur:5:grad_cam:mean
  curves: curves:resnet18:0:defocus_blur:5:grad_cam:blur
  curves: curves:resnet18:0:defocus_blur:5:grad_cam:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:grad_cam:black
  curves: curves:resnet18:0:defocus_blur:5:random:mean
  curves: curves:resnet18:0:defocus_blur:5:random:blur
  curves: curves:resnet18:0:defocus_blur:5:random:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random:black
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:integrated_gradients_rolled:black
  curves: curves:resnet18:0:defocus_blur:5:grad_cam_rolled:mean
  curves: curves:resnet18:0:defocus_blur:5:grad_cam_rolled:blur
  curves: curves:resnet18:0:defocus_blur:5:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:grad_cam_rolled:black
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_2:mean
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_2:blur
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_2:black
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_4:mean
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_4:blur
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_4:black
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_8:mean
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_8:blur
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_8:black
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_16:mean
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_16:blur
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_16:black
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_32:mean
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_32:blur
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:defocus_blur:5:random_lowres_32:black
  curves: curves:resnet18:0:fog:1:integrated_gradients:mean
  curves: curves:resnet18:0:fog:1:integrated_gradients:blur
  curves: curves:resnet18:0:fog:1:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:fog:1:integrated_gradients:black
  curves: curves:resnet18:0:fog:1:grad_cam:mean
  curves: curves:resnet18:0:fog:1:grad_cam:blur
  curves: curves:resnet18:0:fog:1:grad_cam:uniform_noise
  curves: curves:resnet18:0:fog:1:grad_cam:black
  curves: curves:resnet18:0:fog:1:random:mean
  curves: curves:resnet18:0:fog:1:random:blur
  curves: curves:resnet18:0:fog:1:random:uniform_noise
  curves: curves:resnet18:0:fog:1:random:black
  curves: curves:resnet18:0:fog:1:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:fog:1:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:fog:1:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:fog:1:integrated_gradients_rolled:black
  curves: curves:resnet18:0:fog:1:grad_cam_rolled:mean
  curves: curves:resnet18:0:fog:1:grad_cam_rolled:blur
  curves: curves:resnet18:0:fog:1:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:fog:1:grad_cam_rolled:black
  curves: curves:resnet18:0:fog:1:random_lowres_2:mean
  curves: curves:resnet18:0:fog:1:random_lowres_2:blur
  curves: curves:resnet18:0:fog:1:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:fog:1:random_lowres_2:black
  curves: curves:resnet18:0:fog:1:random_lowres_4:mean
  curves: curves:resnet18:0:fog:1:random_lowres_4:blur
  curves: curves:resnet18:0:fog:1:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:fog:1:random_lowres_4:black
  curves: curves:resnet18:0:fog:1:random_lowres_8:mean
  curves: curves:resnet18:0:fog:1:random_lowres_8:blur
  curves: curves:resnet18:0:fog:1:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:fog:1:random_lowres_8:black
  curves: curves:resnet18:0:fog:1:random_lowres_16:mean
  curves: curves:resnet18:0:fog:1:random_lowres_16:blur
  curves: curves:resnet18:0:fog:1:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:fog:1:random_lowres_16:black
  curves: curves:resnet18:0:fog:1:random_lowres_32:mean
  curves: curves:resnet18:0:fog:1:random_lowres_32:blur
  curves: curves:resnet18:0:fog:1:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:fog:1:random_lowres_32:black
  curves: curves:resnet18:0:fog:3:integrated_gradients:mean
  curves: curves:resnet18:0:fog:3:integrated_gradients:blur
  curves: curves:resnet18:0:fog:3:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:fog:3:integrated_gradients:black
  curves: curves:resnet18:0:fog:3:grad_cam:mean
  curves: curves:resnet18:0:fog:3:grad_cam:blur
  curves: curves:resnet18:0:fog:3:grad_cam:uniform_noise
  curves: curves:resnet18:0:fog:3:grad_cam:black
  curves: curves:resnet18:0:fog:3:random:mean
  curves: curves:resnet18:0:fog:3:random:blur
  curves: curves:resnet18:0:fog:3:random:uniform_noise
  curves: curves:resnet18:0:fog:3:random:black
  curves: curves:resnet18:0:fog:3:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:fog:3:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:fog:3:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:fog:3:integrated_gradients_rolled:black
  curves: curves:resnet18:0:fog:3:grad_cam_rolled:mean
  curves: curves:resnet18:0:fog:3:grad_cam_rolled:blur
  curves: curves:resnet18:0:fog:3:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:fog:3:grad_cam_rolled:black
  curves: curves:resnet18:0:fog:3:random_lowres_2:mean
  curves: curves:resnet18:0:fog:3:random_lowres_2:blur
  curves: curves:resnet18:0:fog:3:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:fog:3:random_lowres_2:black
  curves: curves:resnet18:0:fog:3:random_lowres_4:mean
  curves: curves:resnet18:0:fog:3:random_lowres_4:blur
  curves: curves:resnet18:0:fog:3:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:fog:3:random_lowres_4:black
  curves: curves:resnet18:0:fog:3:random_lowres_8:mean
  curves: curves:resnet18:0:fog:3:random_lowres_8:blur
  curves: curves:resnet18:0:fog:3:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:fog:3:random_lowres_8:black
  curves: curves:resnet18:0:fog:3:random_lowres_16:mean
  curves: curves:resnet18:0:fog:3:random_lowres_16:blur
  curves: curves:resnet18:0:fog:3:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:fog:3:random_lowres_16:black
  curves: curves:resnet18:0:fog:3:random_lowres_32:mean
  curves: curves:resnet18:0:fog:3:random_lowres_32:blur
  curves: curves:resnet18:0:fog:3:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:fog:3:random_lowres_32:black
  curves: curves:resnet18:0:fog:5:integrated_gradients:mean
  curves: curves:resnet18:0:fog:5:integrated_gradients:blur
  curves: curves:resnet18:0:fog:5:integrated_gradients:uniform_noise
  curves: curves:resnet18:0:fog:5:integrated_gradients:black
  curves: curves:resnet18:0:fog:5:grad_cam:mean
  curves: curves:resnet18:0:fog:5:grad_cam:blur
  curves: curves:resnet18:0:fog:5:grad_cam:uniform_noise
  curves: curves:resnet18:0:fog:5:grad_cam:black
  curves: curves:resnet18:0:fog:5:random:mean
  curves: curves:resnet18:0:fog:5:random:blur
  curves: curves:resnet18:0:fog:5:random:uniform_noise
  curves: curves:resnet18:0:fog:5:random:black
  curves: curves:resnet18:0:fog:5:integrated_gradients_rolled:mean
  curves: curves:resnet18:0:fog:5:integrated_gradients_rolled:blur
  curves: curves:resnet18:0:fog:5:integrated_gradients_rolled:uniform_noise
  curves: curves:resnet18:0:fog:5:integrated_gradients_rolled:black
  curves: curves:resnet18:0:fog:5:grad_cam_rolled:mean
  curves: curves:resnet18:0:fog:5:grad_cam_rolled:blur
  curves: curves:resnet18:0:fog:5:grad_cam_rolled:uniform_noise
  curves: curves:resnet18:0:fog:5:grad_cam_rolled:black
  curves: curves:resnet18:0:fog:5:random_lowres_2:mean
  curves: curves:resnet18:0:fog:5:random_lowres_2:blur
  curves: curves:resnet18:0:fog:5:random_lowres_2:uniform_noise
  curves: curves:resnet18:0:fog:5:random_lowres_2:black
  curves: curves:resnet18:0:fog:5:random_lowres_4:mean
  curves: curves:resnet18:0:fog:5:random_lowres_4:blur
  curves: curves:resnet18:0:fog:5:random_lowres_4:uniform_noise
  curves: curves:resnet18:0:fog:5:random_lowres_4:black
  curves: curves:resnet18:0:fog:5:random_lowres_8:mean
  curves: curves:resnet18:0:fog:5:random_lowres_8:blur
  curves: curves:resnet18:0:fog:5:random_lowres_8:uniform_noise
  curves: curves:resnet18:0:fog:5:random_lowres_8:black
  curves: curves:resnet18:0:fog:5:random_lowres_16:mean
  curves: curves:resnet18:0:fog:5:random_lowres_16:blur
  curves: curves:resnet18:0:fog:5:random_lowres_16:uniform_noise
  curves: curves:resnet18:0:fog:5:random_lowres_16:black
  curves: curves:resnet18:0:fog:5:random_lowres_32:mean
  curves: curves:resnet18:0:fog:5:random_lowres_32:blur
  curves: curves:resnet18:0:fog:5:random_lowres_32:uniform_noise
  curves: curves:resnet18:0:fog:5:random_lowres_32:black

============================================================
Completed 510 cells, skipped 0 cached, failed 0 cells
Elapsed time: 6746.0s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean, blur, uniform_noise, black
explainers  integrated_gradients, grad_cam, random, integrated_gradients_rolled, grad_cam_rolled, random_lowres_2, random_lowres_4, random_lowres_8, random_lowres_16, random_lowres_32

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8610    0.2166    0.0516    0.0764    0.0315
   1     0.8330    0.2522    0.0619    0.0790    0.0430
   3     0.7247    0.4160    0.1318    0.1495    0.1114
   5     0.5487    0.6810    0.2514    0.2650    0.2702

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================

  imputation: black
    grad_cam               mean -0.29461   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.15436   1/10 cells above 0  
    integrated_gradients   mean -0.02465   1/10 cells above 0  
    integrated_gradients_rolled mean -0.01907   1/10 cells above 0  
    random                 mean -0.00003   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.04089   2/10 cells above 0  
    random_lowres_2        mean -0.22408   1/10 cells above 0  
    random_lowres_32       mean -0.07605   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.15062   1/10 cells above 0  
    random_lowres_8        mean -0.07525   2/10 cells above 0  

  imputation: blur
    grad_cam               mean -0.06140   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean +0.02825   7/10 cells above 0  
    integrated_gradients   mean -0.06773   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean +0.00591   2/10 cells above 0  
    random                 mean -0.00003   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean +0.00398   2/10 cells above 0  
    random_lowres_2        mean +0.02776   7/10 cells above 0  
    random_lowres_32       mean +0.00032   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean +0.02373   7/10 cells above 0  
    random_lowres_8        mean +0.02224   8/10 cells above 0  

  imputation: mean
    grad_cam               mean -0.29145   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.15615   1/10 cells above 0  
    integrated_gradients   mean -0.10945   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.03457   1/10 cells above 0  
    random                 mean -0.00006   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.08005   1/10 cells above 0  
    random_lowres_2        mean -0.20245   1/10 cells above 0  
    random_lowres_32       mean -0.05612   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.14949   1/10 cells above 0  
    random_lowres_8        mean -0.10383   1/10 cells above 0  

  imputation: uniform_noise
    grad_cam               mean -0.29300   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.16298   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients   mean -0.06792   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.02697   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00001   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.08006   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_2        mean -0.20902   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean -0.05737   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.17004   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_8        mean -0.12855   0/10 cells above 0  NO CELL BEATS RANDOM

imputation      explainer               sev   faithfulness   half_width
black           grad_cam                  0       -0.36030      0.01552
black           grad_cam                  1       -0.34085      0.01598
black           grad_cam                  3       -0.27991      0.01720
black           grad_cam                  5       -0.24117      0.01609
black           grad_cam_rolled           0       -0.20760      0.01878
black           grad_cam_rolled           1       -0.19099      0.01949
black           grad_cam_rolled           3       -0.13045      0.02048
black           grad_cam_rolled           5       -0.12389      0.01946
black           integrated_gradients      0       -0.03866      0.00702
black           integrated_gradients      1       -0.03268      0.00672
black           integrated_gradients      3       -0.02339      0.00642
black           integrated_gradients      5       -0.01321      0.00614
black           integrated_gradients_rolled    0       -0.03182      0.00727
black           integrated_gradients_rolled    1       -0.02592      0.00688
black           integrated_gradients_rolled    3       -0.01671      0.00661
black           integrated_gradients_rolled    5       -0.01034      0.00627
black           random                    0       -0.00007      0.00017
black           random                    1       -0.00006      0.00015
black           random                    3       -0.00003      0.00016
black           random                    5       +0.00000      0.00016
black           random_lowres_16          0       -0.07312      0.01226
black           random_lowres_16          1       -0.06325      0.01284
black           random_lowres_16          3       -0.03236      0.01361
black           random_lowres_16          5       -0.01631      0.01207
black           random_lowres_2           0       -0.28497      0.01920
black           random_lowres_2           1       -0.26694      0.01948
black           random_lowres_2           3       -0.20281      0.02107
black           random_lowres_2           5       -0.18217      0.02041
black           random_lowres_32          0       -0.09646      0.00716
black           random_lowres_32          1       -0.09146      0.00711
black           random_lowres_32          3       -0.07467      0.00696
black           random_lowres_32          5       -0.05522      0.00652
black           random_lowres_4           0       -0.20542      0.01874
black           random_lowres_4           1       -0.18838      0.01945
black           random_lowres_4           3       -0.12947      0.02067
black           random_lowres_4           5       -0.11575      0.01962
black           random_lowres_8           0       -0.11885      0.01666
black           random_lowres_8           1       -0.10388      0.01738
black           random_lowres_8           3       -0.05961      0.01860
black           random_lowres_8           5       -0.04772      0.01710
blur            grad_cam                  0       -0.05039      0.00954
blur            grad_cam                  1       -0.05438      0.01028
blur            grad_cam                  3       -0.07131      0.01106
blur            grad_cam                  5       -0.06217      0.01058
blur            grad_cam_rolled           0       +0.02860      0.01050
blur            grad_cam_rolled           1       +0.02820      0.01125
blur            grad_cam_rolled           3       +0.03013      0.01168
blur            grad_cam_rolled           5       +0.02630      0.01150
blur            integrated_gradients      0       -0.05930      0.00773
blur            integrated_gradients      1       -0.06123      0.00837
blur            integrated_gradients      3       -0.07636      0.00908
blur            integrated_gradients      5       -0.06840      0.00913
blur            integrated_gradients_rolled    0       +0.00913      0.00809
blur            integrated_gradients_rolled    1       +0.00579      0.00854
blur            integrated_gradients_rolled    3       +0.00583      0.00873
blur            integrated_gradients_rolled    5       +0.00505      0.00860
blur            random                    0       -0.00001      0.00007
blur            random                    1       -0.00005      0.00008
blur            random                    3       -0.00004      0.00010
blur            random                    5       +0.00001      0.00008
blur            random_lowres_16          0       +0.00192      0.00860
blur            random_lowres_16          1       +0.00154      0.00891
blur            random_lowres_16          3       +0.00471      0.00946
blur            random_lowres_16          5       +0.00638      0.00950
blur            random_lowres_2           0       +0.03327      0.00966
blur            random_lowres_2           1       +0.03166      0.01003
blur            random_lowres_2           3       +0.02885      0.01089
blur            random_lowres_2           5       +0.02094      0.01090
blur            random_lowres_32          0       -0.00618      0.00662
blur            random_lowres_32          1       -0.00039      0.00666
blur            random_lowres_32          3       +0.00335      0.00714
blur            random_lowres_32          5       +0.00015      0.00732
blur            random_lowres_4           0       +0.02386      0.01067
blur            random_lowres_4           1       +0.02622      0.01111
blur            random_lowres_4           3       +0.02435      0.01131
blur            random_lowres_4           5       +0.02057      0.01098
blur            random_lowres_8           0       +0.02305      0.00994
blur            random_lowres_8           1       +0.02313      0.01013
blur            random_lowres_8           3       +0.02445      0.01067
blur            random_lowres_8           5       +0.01887      0.01068
mean            grad_cam                  0       -0.35192      0.01655
mean            grad_cam                  1       -0.33593      0.01708
mean            grad_cam                  3       -0.27870      0.01849
mean            grad_cam                  5       -0.23956      0.01785
mean            grad_cam_rolled           0       -0.21019      0.01842
mean            grad_cam_rolled           1       -0.19707      0.01893
mean            grad_cam_rolled           3       -0.13659      0.01998
mean            grad_cam_rolled           5       -0.11678      0.01923
mean            integrated_gradients      0       -0.13794      0.00934
mean            integrated_gradients      1       -0.12986      0.00915
mean            integrated_gradients      3       -0.10688      0.00889
mean            integrated_gradients      5       -0.08209      0.00822
mean            integrated_gradients_rolled    0       -0.04796      0.00810
mean            integrated_gradients_rolled    1       -0.04458      0.00829
mean            integrated_gradients_rolled    3       -0.03084      0.00813
mean            integrated_gradients_rolled    5       -0.02383      0.00747
mean            random                    0       -0.00009      0.00012
mean            random                    1       -0.00006      0.00014
mean            random                    3       -0.00007      0.00013
mean            random                    5       -0.00004      0.00012
mean            random_lowres_16          0       -0.11449      0.01337
mean            random_lowres_16          1       -0.10419      0.01314
mean            random_lowres_16          3       -0.07176      0.01346
mean            random_lowres_16          5       -0.05271      0.01284
mean            random_lowres_2           0       -0.25468      0.01773
mean            random_lowres_2           1       -0.24306      0.01805
mean            random_lowres_2           3       -0.18683      0.01939
mean            random_lowres_2           5       -0.16007      0.01877
mean            random_lowres_32          0       -0.07069      0.00697
mean            random_lowres_32          1       -0.06755      0.00710
mean            random_lowres_32          3       -0.05248      0.00735
mean            random_lowres_32          5       -0.04346      0.00699
mean            random_lowres_4           0       -0.19837      0.01874
mean            random_lowres_4           1       -0.18441      0.01867
mean            random_lowres_4           3       -0.13457      0.01998
mean            random_lowres_4           5       -0.11320      0.01905
mean            random_lowres_8           0       -0.14639      0.01756
mean            random_lowres_8           1       -0.13628      0.01756
mean            random_lowres_8           3       -0.09211      0.01778
mean            random_lowres_8           5       -0.06890      0.01704
uniform_noise   grad_cam                  0       -0.34995      0.01365
uniform_noise   grad_cam                  1       -0.33525      0.01392
uniform_noise   grad_cam                  3       -0.28405      0.01430
uniform_noise   grad_cam                  5       -0.24071      0.01360
uniform_noise   grad_cam_rolled           0       -0.20835      0.01308
uniform_noise   grad_cam_rolled           1       -0.19652      0.01343
uniform_noise   grad_cam_rolled           3       -0.15301      0.01322
uniform_noise   grad_cam_rolled           5       -0.12429      0.01200
uniform_noise   integrated_gradients      0       -0.09134      0.00619
uniform_noise   integrated_gradients      1       -0.08398      0.00628
uniform_noise   integrated_gradients      3       -0.06405      0.00625
uniform_noise   integrated_gradients      5       -0.04793      0.00594
uniform_noise   integrated_gradients_rolled    0       -0.03547      0.00566
uniform_noise   integrated_gradients_rolled    1       -0.03267      0.00546
uniform_noise   integrated_gradients_rolled    3       -0.02582      0.00541
uniform_noise   integrated_gradients_rolled    5       -0.01961      0.00482
uniform_noise   random                    0       -0.00002      0.00009
uniform_noise   random                    1       -0.00001      0.00008
uniform_noise   random                    3       -0.00002      0.00005
uniform_noise   random                    5       -0.00001      0.00005
uniform_noise   random_lowres_16          0       -0.10430      0.00787
uniform_noise   random_lowres_16          1       -0.09762      0.00750
uniform_noise   random_lowres_16          3       -0.07637      0.00773
uniform_noise   random_lowres_16          5       -0.05812      0.00688
uniform_noise   random_lowres_2           0       -0.26107      0.01316
uniform_noise   random_lowres_2           1       -0.24644      0.01310
uniform_noise   random_lowres_2           3       -0.19934      0.01344
uniform_noise   random_lowres_2           5       -0.16393      0.01264
uniform_noise   random_lowres_32          0       -0.07327      0.00621
uniform_noise   random_lowres_32          1       -0.06844      0.00607
uniform_noise   random_lowres_32          3       -0.05471      0.00607
uniform_noise   random_lowres_32          5       -0.04367      0.00562
uniform_noise   random_lowres_4           0       -0.21873      0.01228
uniform_noise   random_lowres_4           1       -0.20530      0.01279
uniform_noise   random_lowres_4           3       -0.15933      0.01277
uniform_noise   random_lowres_4           5       -0.12924      0.01184
uniform_noise   random_lowres_8           0       -0.16308      0.01070
uniform_noise   random_lowres_8           1       -0.15394      0.01076
uniform_noise   random_lowres_8           3       -0.12175      0.01069
uniform_noise   random_lowres_8           5       -0.09846      0.01003

========================================================================
GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid
                  lattice, upsampled as Grad-CAM is. No model input.
========================================================================
imputation       grid   faithfulness   half_width
black               2       -0.22408      0.02021
black               4       -0.15062      0.01980
black               8       -0.07525      0.01759
black              16       -0.04089      0.01278
black              32       -0.07605      0.00689
blur                2       +0.02776      0.01051
blur                4       +0.02373      0.01109
blur                8       +0.02224      0.01043
blur               16       +0.00398      0.00922
blur               32       +0.00032      0.00700
mean                2       -0.20245      0.01863
mean                4       -0.14949      0.01918
mean                8       -0.10383      0.01747
mean               16       -0.08005      0.01317
mean               32       -0.05612      0.00713
uniform_noise       2       -0.20902      0.01307
uniform_noise       4       -0.17004      0.01245
uniform_noise       8       -0.12855      0.01051
uniform_noise      16       -0.08006      0.00742
uniform_noise      32       -0.05737      0.00595

Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on this
curve at grid=4, its score is its resolution and not its content.

========================================================================
GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the
                                same mask geometry; >0 beats it
========================================================================
imputation      explainer                vs                   kind          matched     half_w  verdict
black           grad_cam                 grad_cam_rolled      alignment    -0.14025    0.01396  NO CELL BEATS ITS MATCHED CONTROL
black           grad_cam                 random_lowres_4      geometry     -0.14399    0.01437  NO CELL BEATS ITS MATCHED CONTROL
black           integrated_gradients     integrated_gradients_rolled alignment    -0.00557    0.00596  NO CELL BEATS ITS MATCHED CONTROL
blur            grad_cam                 grad_cam_rolled      alignment    -0.08964    0.01068  NO CELL BEATS ITS MATCHED CONTROL
blur            grad_cam                 random_lowres_4      geometry     -0.08513    0.01021  NO CELL BEATS ITS MATCHED CONTROL
blur            integrated_gradients     integrated_gradients_rolled alignment    -0.07364    0.00940  NO CELL BEATS ITS MATCHED CONTROL
mean            grad_cam                 grad_cam_rolled      alignment    -0.13530    0.01392  NO CELL BEATS ITS MATCHED CONTROL
mean            grad_cam                 random_lowres_4      geometry     -0.14196    0.01423  NO CELL BEATS ITS MATCHED CONTROL
mean            integrated_gradients     integrated_gradients_rolled alignment    -0.07488    0.00879  NO CELL BEATS ITS MATCHED CONTROL
uniform_noise   grad_cam                 grad_cam_rolled      alignment    -0.13001    0.01188  NO CELL BEATS ITS MATCHED CONTROL
uniform_noise   grad_cam                 random_lowres_4      geometry     -0.12296    0.01181  NO CELL BEATS ITS MATCHED CONTROL
uniform_noise   integrated_gradients     integrated_gradients_rolled alignment    -0.04095    0.00639  NO CELL BEATS ITS MATCHED CONTROL

The pixel-i.i.d. control cancels exactly in this difference: both terms
use the same control_seed over the same images, so what is left is the
explainer scored against its own geometry.

========================================================================
P3 GATE   half-width < 10% of the clean-to-severity-5 change
========================================================================
imputation      explainer              family               change     half_w     ratio  verdict
black           grad_cam               defocus_blur       +0.04443    0.01552     0.349  FAIL
black           grad_cam               fog                +0.08713    0.01552     0.178  FAIL
black           grad_cam               gaussian_noise     +0.22583    0.01904     0.084  PASS
black           grad_cam_rolled        defocus_blur       -0.03349    0.01878     0.561  FAIL
black           grad_cam_rolled        fog                +0.01148    0.01878       n/a  no change
black           grad_cam_rolled        gaussian_noise     +0.27313    0.02483     0.091  PASS
black           integrated_gradients   defocus_blur       +0.00709    0.00702     0.990  FAIL
black           integrated_gradients   fog                +0.01312    0.00702     0.535  FAIL
black           integrated_gradients   gaussian_noise     +0.05614    0.00746     0.133  FAIL
black           integrated_gradients_rolled defocus_blur       +0.00014    0.00727       n/a  no change
black           integrated_gradients_rolled fog                +0.00789    0.00727     0.921  FAIL
black           integrated_gradients_rolled gaussian_noise     +0.05638    0.00773     0.137  FAIL
black           random                 defocus_blur       +0.00013    0.00017       n/a  no change
black           random                 fog                +0.00013    0.00017       n/a  no change
black           random                 gaussian_noise     -0.00004    0.00023       n/a  no change
black           random_lowres_16       defocus_blur       -0.00293    0.01226       n/a  no change
black           random_lowres_16       fog                +0.02275    0.01226     0.539  FAIL
black           random_lowres_16       gaussian_noise     +0.15061    0.01619     0.107  FAIL
black           random_lowres_2        defocus_blur       -0.02149    0.01920     0.894  FAIL
black           random_lowres_2        fog                +0.01892    0.01920       n/a  no change
black           random_lowres_2        gaussian_noise     +0.31094    0.02525     0.081  PASS
black           random_lowres_32       defocus_blur       +0.02878    0.00716     0.249  FAIL
black           random_lowres_32       fog                +0.02906    0.00716     0.246  FAIL
black           random_lowres_32       gaussian_noise     +0.06588    0.00803     0.122  FAIL
black           random_lowres_4        defocus_blur       -0.01927    0.01874     0.972  FAIL
black           random_lowres_4        fog                +0.01601    0.01874       n/a  no change
black           random_lowres_4        gaussian_noise     +0.27227    0.02583     0.095  PASS
black           random_lowres_8        defocus_blur       -0.03008    0.01666     0.554  FAIL
black           random_lowres_8        fog                +0.00768    0.01666       n/a  no change
black           random_lowres_8        gaussian_noise     +0.23580    0.02396     0.102  FAIL
blur            grad_cam               defocus_blur       -0.00339    0.00954       n/a  no change
blur            grad_cam               fog                -0.04988    0.01053     0.211  FAIL
blur            grad_cam               gaussian_noise     +0.01792    0.01250     0.697  FAIL
blur            grad_cam_rolled        defocus_blur       -0.03381    0.01050     0.311  FAIL
blur            grad_cam_rolled        fog                -0.02768    0.01050     0.379  FAIL
blur            grad_cam_rolled        gaussian_noise     +0.05458    0.01475     0.270  FAIL
blur            integrated_gradients   defocus_blur       -0.00766    0.00799       n/a  no change
blur            integrated_gradients   fog                -0.04866    0.00921     0.189  FAIL
blur            integrated_gradients   gaussian_noise     +0.02902    0.01019     0.351  FAIL
blur            integrated_gradients_rolled defocus_blur       -0.00737    0.00809       n/a  no change
blur            integrated_gradients_rolled fog                -0.01333    0.00860     0.645  FAIL
blur            integrated_gradients_rolled gaussian_noise     +0.00845    0.01076       n/a  no change
blur            random                 defocus_blur       -0.00001    0.00007       n/a  no change
blur            random                 fog                +0.00001    0.00008       n/a  no change
blur            random                 gaussian_noise     +0.00006    0.00011       n/a  no change
blur            random_lowres_16       defocus_blur       -0.00636    0.00860       n/a  no change
blur            random_lowres_16       fog                -0.00692    0.00860       n/a  no change
blur            random_lowres_16       gaussian_noise     +0.02665    0.01223     0.459  FAIL
blur            random_lowres_2        defocus_blur       -0.04024    0.00966     0.240  FAIL
blur            random_lowres_2        fog                -0.04081    0.00988     0.242  FAIL
blur            random_lowres_2        gaussian_noise     +0.04408    0.01362     0.309  FAIL
blur            random_lowres_32       defocus_blur       +0.00398    0.00662       n/a  no change
blur            random_lowres_32       fog                +0.00554    0.00690       n/a  no change
blur            random_lowres_32       gaussian_noise     +0.00945    0.00897     0.949  FAIL
blur            random_lowres_4        defocus_blur       -0.02829    0.01067     0.377  FAIL
blur            random_lowres_4        fog                -0.03173    0.01067     0.336  FAIL
blur            random_lowres_4        gaussian_noise     +0.05015    0.01383     0.276  FAIL
blur            random_lowres_8        defocus_blur       -0.02753    0.00994     0.361  FAIL
blur            random_lowres_8        fog                -0.02638    0.01024     0.388  FAIL
blur            random_lowres_8        gaussian_noise     +0.04136    0.01308     0.316  FAIL
mean            grad_cam               defocus_blur       +0.03556    0.01655     0.466  FAIL
mean            grad_cam               fog                +0.04796    0.01655     0.345  FAIL
mean            grad_cam               gaussian_noise     +0.25354    0.02146     0.085  PASS
mean            grad_cam_rolled        defocus_blur       -0.00453    0.01842       n/a  no change
mean            grad_cam_rolled        fog                +0.01156    0.01842       n/a  no change
mean            grad_cam_rolled        gaussian_noise     +0.27319    0.02503     0.092  PASS
mean            integrated_gradients   defocus_blur       +0.05013    0.00934     0.186  FAIL
mean            integrated_gradients   fog                +0.03502    0.00934     0.267  FAIL
mean            integrated_gradients   gaussian_noise     +0.08239    0.00934     0.113  FAIL
mean            integrated_gradients_rolled defocus_blur       +0.00890    0.00810     0.911  FAIL
mean            integrated_gradients_rolled fog                +0.00653    0.00810       n/a  no change
mean            integrated_gradients_rolled gaussian_noise     +0.05697    0.00810     0.142  FAIL
mean            random                 defocus_blur       -0.00000    0.00012       n/a  no change
mean            random                 fog                +0.00001    0.00012       n/a  no change
mean            random                 gaussian_noise     +0.00013    0.00014       n/a  no change
mean            random_lowres_16       defocus_blur       +0.00846    0.01337       n/a  no change
mean            random_lowres_16       fog                +0.02187    0.01337     0.612  FAIL
mean            random_lowres_16       gaussian_noise     +0.15500    0.01637     0.106  FAIL
mean            random_lowres_2        defocus_blur       -0.00740    0.01773       n/a  no change
mean            random_lowres_2        fog                +0.01055    0.01773       n/a  no change
mean            random_lowres_2        gaussian_noise     +0.28069    0.02466     0.088  PASS
mean            random_lowres_32       defocus_blur       +0.02841    0.00697     0.245  FAIL
mean            random_lowres_32       fog                +0.01008    0.00723     0.718  FAIL
mean            random_lowres_32       gaussian_noise     +0.04317    0.00697     0.161  FAIL
mean            random_lowres_4        defocus_blur       -0.00719    0.01874       n/a  no change
mean            random_lowres_4        fog                +0.00026    0.01874       n/a  no change
mean            random_lowres_4        gaussian_noise     +0.26243    0.02545     0.097  PASS
mean            random_lowres_8        defocus_blur       -0.00403    0.01756       n/a  no change
mean            random_lowres_8        fog                +0.00895    0.01756       n/a  no change
mean            random_lowres_8        gaussian_noise     +0.22755    0.02295     0.101  FAIL
uniform_noise   grad_cam               defocus_blur       +0.06854    0.01365     0.199  FAIL
uniform_noise   grad_cam               fog                +0.06810    0.01365     0.200  FAIL
uniform_noise   grad_cam               gaussian_noise     +0.19109    0.01490     0.078  PASS
uniform_noise   grad_cam_rolled        defocus_blur       +0.04840    0.01308     0.270  FAIL
uniform_noise   grad_cam_rolled        fog                +0.05586    0.01308     0.234  FAIL
uniform_noise   grad_cam_rolled        gaussian_noise     +0.14791    0.01364     0.092  PASS
uniform_noise   integrated_gradients   defocus_blur       +0.04255    0.00619     0.145  FAIL
uniform_noise   integrated_gradients   fog                +0.04286    0.00619     0.144  FAIL
uniform_noise   integrated_gradients   gaussian_noise     +0.04483    0.00641     0.143  FAIL
uniform_noise   integrated_gradients_rolled defocus_blur       +0.00892    0.00566     0.635  FAIL
uniform_noise   integrated_gradients_rolled fog                +0.01070    0.00566     0.529  FAIL
uniform_noise   integrated_gradients_rolled gaussian_noise     +0.02796    0.00566     0.202  FAIL
uniform_noise   random                 defocus_blur       +0.00000    0.00009       n/a  no change
uniform_noise   random                 fog                +0.00003    0.00009       n/a  no change
uniform_noise   random                 gaussian_noise     +0.00001    0.00009       n/a  no change
uniform_noise   random_lowres_16       defocus_blur       +0.02644    0.00787     0.298  FAIL
uniform_noise   random_lowres_16       fog                +0.03856    0.00787     0.204  FAIL
uniform_noise   random_lowres_16       gaussian_noise     +0.07352    0.00787     0.107  FAIL
uniform_noise   random_lowres_2        defocus_blur       +0.06161    0.01316     0.214  FAIL
uniform_noise   random_lowres_2        fog                +0.05676    0.01316     0.232  FAIL
uniform_noise   random_lowres_2        gaussian_noise     +0.17306    0.01401     0.081  PASS
uniform_noise   random_lowres_32       defocus_blur       +0.02814    0.00621     0.221  FAIL
uniform_noise   random_lowres_32       fog                +0.01558    0.00621     0.398  FAIL
uniform_noise   random_lowres_32       gaussian_noise     +0.04509    0.00621     0.138  FAIL
uniform_noise   random_lowres_4        defocus_blur       +0.06190    0.01228     0.198  FAIL
uniform_noise   random_lowres_4        fog                +0.05884    0.01228     0.209  FAIL
uniform_noise   random_lowres_4        gaussian_noise     +0.14774    0.01302     0.088  PASS
uniform_noise   random_lowres_8        defocus_blur       +0.03464    0.01070     0.309  FAIL
uniform_noise   random_lowres_8        fog                +0.04535    0.01070     0.236  FAIL
uniform_noise   random_lowres_8        gaussian_noise     +0.11388    0.01131     0.099  PASS

GATE: 13/84 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================

(end of Part 1 — everything above this line is VOID)

---

## Part 2 — VALID: the corrected run under `curves-v3`

**2026-10-15. This is the run to cite.** The inverted-mask defect described in
Part 1 is fixed: `curves._get_removal_mask` now selects the HIGHEST-magnitude
pixels, pinned by tests against a map whose maximum and minimum are in known
places. The whole P3b validity grid was recomputed under `curves-v3` -- 510
units, 0 failed, 7325.1s -- on `resnet18`, seed 0, 1000 images, 10 cells
(clean + 3 families x 3 severities), 4 imputation schemes and 10 explainer
arms. Nothing here is shared with Part 1; it is a different computation over
the same images and the same attributions.

Three things to read first, before the raw tables:

1. **FAITHFULNESS** below is the RAW metric, against the pre-registered
   per-pixel-i.i.d. control. It is not a valid level: the GEOMETRY LADDER two
   sections down shows a model-free coarse blob reading -0.19932 under `black`,
   so the mask-shape artefact is as large as any explainer effect measured here.
2. **GEOMETRY-MATCHED FAITHFULNESS** is the valid comparison -- explainer minus
   a control of the same mask geometry. It is positive in 11 of 12 comparisons
   at 7.3 to 10.4 half-widths. The one exception, `black`/IG at -0.00398 and 0.6
   half-widths, is unresolved: not a pass and not a failure.
3. **The P3 GATE at 14/85 is computed on the raw metric**, and 12 of its 14
   passes are control arms -- 9 model-free `random_lowres_*`, 3 `grad_cam_rolled`
   -- with only `black`/grad_cam and `mean`/grad_cam involving a correctly-aimed
   explainer. All 14 are `gaussian_noise`; `blur` passes nothing. A
   matched-metric gate is being added separately.

The calibration block is identical to every prior run, void and valid alike: it
is computed from predictions and reads no curve.

Single seed throughout: `resnet18`, seed 0, 1000 images.

============================================================
Completed 510 cells, skipped 0 cached, failed 0 cells
Elapsed time: 7325.1s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean, blur, uniform_noise, black
explainers  integrated_gradients, grad_cam, random, integrated_gradients_rolled, grad_cam_rolled, random_lowres_2, random_lowres_4, random_lowres_8, random_lowres_16, random_lowres_32

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8610    0.2166    0.0516    0.0764    0.0315
   1     0.8330    0.2522    0.0619    0.0790    0.0430
   3     0.7247    0.4160    0.1318    0.1495    0.1114
   5     0.5487    0.6810    0.2514    0.2650    0.2702

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================

  imputation: black
    grad_cam               mean -0.04735   2/10 cells above 0  
    grad_cam_rolled        mean -0.14802   1/10 cells above 0  
    integrated_gradients   mean -0.04214   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.03817   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.02958   2/10 cells above 0  
    random_lowres_2        mean -0.19932   1/10 cells above 0  
    random_lowres_32       mean +0.00531   3/10 cells above 0  
    random_lowres_4        mean -0.13929   1/10 cells above 0  
    random_lowres_8        mean -0.07275   2/10 cells above 0  

  imputation: blur
    grad_cam               mean +0.13321   10/10 cells above 0  
    grad_cam_rolled        mean +0.03673   9/10 cells above 0  
    integrated_gradients   mean +0.07246   10/10 cells above 0  
    integrated_gradients_rolled mean +0.00146   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean +0.00974   5/10 cells above 0  
    random_lowres_2        mean +0.04398   9/10 cells above 0  
    random_lowres_32       mean +0.01548   9/10 cells above 0  
    random_lowres_4        mean +0.04112   9/10 cells above 0  
    random_lowres_8        mean +0.02673   9/10 cells above 0  

  imputation: mean
    grad_cam               mean -0.02421   2/10 cells above 0  
    grad_cam_rolled        mean -0.13944   1/10 cells above 0  
    integrated_gradients   mean +0.02605   9/10 cells above 0  
    integrated_gradients_rolled mean -0.04784   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00007   3/10 cells above 0  
    random_lowres_16       mean -0.07262   1/10 cells above 0  
    random_lowres_2        mean -0.15983   1/10 cells above 0  
    random_lowres_32       mean +0.00164   2/10 cells above 0  
    random_lowres_4        mean -0.12500   1/10 cells above 0  
    random_lowres_8        mean -0.09769   2/10 cells above 0  

  imputation: uniform_noise
    grad_cam               mean -0.04382   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.16466   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients   mean +0.00203   3/10 cells above 0  
    integrated_gradients_rolled mean -0.04652   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.06960   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_2        mean -0.18550   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean -0.00027   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.15432   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_8        mean -0.11924   0/10 cells above 0  NO CELL BEATS RANDOM

imputation      explainer               sev   faithfulness   half_width
black           grad_cam                  0       -0.07593      0.01897
black           grad_cam                  1       -0.06058      0.01979
black           grad_cam                  3       -0.02262      0.02042
black           grad_cam                  5       -0.04932      0.01969
black           grad_cam_rolled           0       -0.20265      0.01957
black           grad_cam_rolled           1       -0.18291      0.02040
black           grad_cam_rolled           3       -0.12530      0.02105
black           grad_cam_rolled           5       -0.11765      0.02001
black           integrated_gradients      0       -0.06139      0.00793
black           integrated_gradients      1       -0.05478      0.00796
black           integrated_gradients      3       -0.03783      0.00750
black           integrated_gradients      5       -0.02741      0.00651
black           integrated_gradients_rolled    0       -0.05671      0.00738
black           integrated_gradients_rolled    1       -0.04735      0.00761
black           integrated_gradients_rolled    3       -0.03441      0.00732
black           integrated_gradients_rolled    5       -0.02656      0.00676
black           random                    0       +0.00002      0.00012
black           random                    1       +0.00004      0.00012
black           random                    3       +0.00001      0.00012
black           random                    5       +0.00001      0.00011
black           random_lowres_16          0       -0.06017      0.01211
black           random_lowres_16          1       -0.04876      0.01250
black           random_lowres_16          3       -0.01853      0.01308
black           random_lowres_16          5       -0.01126      0.01179
black           random_lowres_2           0       -0.24898      0.01936
black           random_lowres_2           1       -0.23352      0.02001
black           random_lowres_2           3       -0.17614      0.02126
black           random_lowres_2           5       -0.17173      0.02023
black           random_lowres_32          0       +0.00216      0.00519
black           random_lowres_32          1       +0.00406      0.00502
black           random_lowres_32          3       +0.00752      0.00491
black           random_lowres_32          5       +0.00541      0.00451
black           random_lowres_4           0       -0.19355      0.01920
black           random_lowres_4           1       -0.17360      0.01963
black           random_lowres_4           3       -0.11586      0.02037
black           random_lowres_4           5       -0.11032      0.01923
black           random_lowres_8           0       -0.12366      0.01712
black           random_lowres_8           1       -0.10552      0.01796
black           random_lowres_8           3       -0.05350      0.01886
black           random_lowres_8           5       -0.04225      0.01675
blur            grad_cam                  0       +0.13101      0.01203
blur            grad_cam                  1       +0.13677      0.01263
blur            grad_cam                  3       +0.14379      0.01271
blur            grad_cam                  5       +0.11980      0.01226
blur            grad_cam_rolled           0       +0.03306      0.01039
blur            grad_cam_rolled           1       +0.03763      0.01075
blur            grad_cam_rolled           3       +0.03982      0.01185
blur            grad_cam_rolled           5       +0.03397      0.01113
blur            integrated_gradients      0       +0.06346      0.00850
blur            integrated_gradients      1       +0.06672      0.00910
blur            integrated_gradients      3       +0.08385      0.00942
blur            integrated_gradients      5       +0.06982      0.00946
blur            integrated_gradients_rolled    0       +0.00019      0.00730
blur            integrated_gradients_rolled    1       +0.00188      0.00792
blur            integrated_gradients_rolled    3       +0.00363      0.00876
blur            integrated_gradients_rolled    5       -0.00072      0.00834
blur            random                    0       +0.00004      0.00010
blur            random                    1       +0.00006      0.00011
blur            random                    3       -0.00004      0.00012
blur            random                    5       +0.00001      0.00015
blur            random_lowres_16          0       +0.00992      0.00797
blur            random_lowres_16          1       +0.00945      0.00848
blur            random_lowres_16          3       +0.00939      0.00914
blur            random_lowres_16          5       +0.01032      0.00947
blur            random_lowres_2           0       +0.04713      0.00983
blur            random_lowres_2           1       +0.04785      0.01054
blur            random_lowres_2           3       +0.04528      0.01112
blur            random_lowres_2           5       +0.03777      0.01078
blur            random_lowres_32          0       +0.02070      0.00693
blur            random_lowres_32          1       +0.02112      0.00714
blur            random_lowres_32          3       +0.01343      0.00715
blur            random_lowres_32          5       +0.01015      0.00754
blur            random_lowres_4           0       +0.04006      0.00991
blur            random_lowres_4           1       +0.04183      0.01114
blur            random_lowres_4           3       +0.04494      0.01120
blur            random_lowres_4           5       +0.03694      0.01123
blur            random_lowres_8           0       +0.02293      0.00947
blur            random_lowres_8           1       +0.02452      0.01017
blur            random_lowres_8           3       +0.02922      0.01052
blur            random_lowres_8           5       +0.02772      0.01055
mean            grad_cam                  0       -0.05815      0.01716
mean            grad_cam                  1       -0.04436      0.01726
mean            grad_cam                  3       -0.00829      0.01779
mean            grad_cam                  5       -0.00867      0.01677
mean            grad_cam_rolled           0       -0.19263      0.01874
mean            grad_cam_rolled           1       -0.17750      0.01932
mean            grad_cam_rolled           3       -0.12531      0.02006
mean            grad_cam_rolled           5       -0.09780      0.01897
mean            integrated_gradients      0       +0.02810      0.00699
mean            integrated_gradients      1       +0.03020      0.00730
mean            integrated_gradients      3       +0.02883      0.00715
mean            integrated_gradients      5       +0.01842      0.00703
mean            integrated_gradients_rolled    0       -0.06057      0.00801
mean            integrated_gradients_rolled    1       -0.05669      0.00802
mean            integrated_gradients_rolled    3       -0.04377      0.00813
mean            integrated_gradients_rolled    5       -0.03882      0.00758
mean            random                    0       +0.00017      0.00016
mean            random                    1       +0.00014      0.00015
mean            random                    3       +0.00003      0.00015
mean            random                    5       +0.00002      0.00015
mean            random_lowres_16          0       -0.10768      0.01323
mean            random_lowres_16          1       -0.09647      0.01313
mean            random_lowres_16          3       -0.06305      0.01368
mean            random_lowres_16          5       -0.04666      0.01261
mean            random_lowres_2           0       -0.20590      0.01750
mean            random_lowres_2           1       -0.19426      0.01822
mean            random_lowres_2           3       -0.14401      0.01861
mean            random_lowres_2           5       -0.12587      0.01796
mean            random_lowres_32          0       -0.00309      0.00630
mean            random_lowres_32          1       -0.00080      0.00619
mean            random_lowres_32          3       +0.00300      0.00636
mean            random_lowres_32          5       +0.00431      0.00602
mean            random_lowres_4           0       -0.17507      0.01843
mean            random_lowres_4           1       -0.16162      0.01853
mean            random_lowres_4           3       -0.10952      0.01965
mean            random_lowres_4           5       -0.08715      0.01821
mean            random_lowres_8           0       -0.14582      0.01715
mean            random_lowres_8           1       -0.13206      0.01735
mean            random_lowres_8           3       -0.08495      0.01773
mean            random_lowres_8           5       -0.06000      0.01688
uniform_noise   grad_cam                  0       -0.06842      0.00893
uniform_noise   grad_cam                  1       -0.05924      0.00897
uniform_noise   grad_cam                  3       -0.03737      0.00809
uniform_noise   grad_cam                  5       -0.02665      0.00789
uniform_noise   grad_cam_rolled           0       -0.21145      0.01358
uniform_noise   grad_cam_rolled           1       -0.19869      0.01366
uniform_noise   grad_cam_rolled           3       -0.15465      0.01314
uniform_noise   grad_cam_rolled           5       -0.12502      0.01230
uniform_noise   integrated_gradients      0       +0.00260      0.00484
uniform_noise   integrated_gradients      1       +0.00437      0.00504
uniform_noise   integrated_gradients      3       +0.00328      0.00485
uniform_noise   integrated_gradients      5       -0.00175      0.00487
uniform_noise   integrated_gradients_rolled    0       -0.06007      0.00593
uniform_noise   integrated_gradients_rolled    1       -0.05529      0.00605
uniform_noise   integrated_gradients_rolled    3       -0.04184      0.00595
uniform_noise   integrated_gradients_rolled    5       -0.03791      0.00550
uniform_noise   random                    0       +0.00008      0.00011
uniform_noise   random                    1       +0.00009      0.00012
uniform_noise   random                    3       -0.00003      0.00012
uniform_noise   random                    5       -0.00001      0.00011
uniform_noise   random_lowres_16          0       -0.09149      0.00809
uniform_noise   random_lowres_16          1       -0.08548      0.00808
uniform_noise   random_lowres_16          3       -0.06471      0.00788
uniform_noise   random_lowres_16          5       -0.05132      0.00708
uniform_noise   random_lowres_2           0       -0.23212      0.01281
uniform_noise   random_lowres_2           1       -0.21970      0.01313
uniform_noise   random_lowres_2           3       -0.17493      0.01289
uniform_noise   random_lowres_2           5       -0.14633      0.01187
uniform_noise   random_lowres_32          0       -0.00065      0.00361
uniform_noise   random_lowres_32          1       -0.00047      0.00347
uniform_noise   random_lowres_32          3       +0.00057      0.00346
uniform_noise   random_lowres_32          5       -0.00080      0.00325
uniform_noise   random_lowres_4           0       -0.19989      0.01256
uniform_noise   random_lowres_4           1       -0.18862      0.01252
uniform_noise   random_lowres_4           3       -0.14404      0.01266
uniform_noise   random_lowres_4           5       -0.11512      0.01150
uniform_noise   random_lowres_8           0       -0.15836      0.01062
uniform_noise   random_lowres_8           1       -0.14728      0.01053
uniform_noise   random_lowres_8           3       -0.10984      0.01060
uniform_noise   random_lowres_8           5       -0.08755      0.00966

========================================================================
GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid
                  lattice, upsampled as Grad-CAM is. No model input.
========================================================================
imputation       grid   faithfulness   half_width
black               2       -0.19932      0.02039
black               4       -0.13929      0.01969
black               8       -0.07275      0.01778
black              16       -0.02958      0.01242
black              32       +0.00531      0.00485
blur                2       +0.04398      0.01071
blur                4       +0.04112      0.01106
blur                8       +0.02673      0.01032
blur               16       +0.00974      0.00892
blur               32       +0.01548      0.00724
mean                2       -0.15983      0.01819
mean                4       -0.12500      0.01876
mean                8       -0.09769      0.01730
mean               16       -0.07262      0.01315
mean               32       +0.00164      0.00620
uniform_noise       2       -0.18550      0.01265
uniform_noise       4       -0.15432      0.01226
uniform_noise       8       -0.11924      0.01030
uniform_noise      16       -0.06960      0.00772
uniform_noise      32       -0.00027      0.00341

Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on this
curve at grid=4, its score is its resolution and not its content.

========================================================================
GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the
                                same mask geometry; >0 beats it
========================================================================
imputation      explainer                vs                   kind          matched     half_w  verdict
black           grad_cam                 grad_cam_rolled      alignment    +0.10068    0.01169  10/10 cells beat it
black           grad_cam                 random_lowres_4      geometry     +0.09194    0.01088  10/10 cells beat it
black           integrated_gradients     integrated_gradients_rolled alignment    -0.00398    0.00638  1/10 cells beat it
blur            grad_cam                 grad_cam_rolled      alignment    +0.09648    0.01107  10/10 cells beat it
blur            grad_cam                 random_lowres_4      geometry     +0.09209    0.01046  10/10 cells beat it
blur            integrated_gradients     integrated_gradients_rolled alignment    +0.07100    0.00973  10/10 cells beat it
mean            grad_cam                 grad_cam_rolled      alignment    +0.11523    0.01233  10/10 cells beat it
mean            grad_cam                 random_lowres_4      geometry     +0.10078    0.01160  10/10 cells beat it
mean            integrated_gradients     integrated_gradients_rolled alignment    +0.07389    0.00810  10/10 cells beat it
uniform_noise   grad_cam                 grad_cam_rolled      alignment    +0.12083    0.01160  10/10 cells beat it
uniform_noise   grad_cam                 random_lowres_4      geometry     +0.11050    0.01085  10/10 cells beat it
uniform_noise   integrated_gradients     integrated_gradients_rolled alignment    +0.04855    0.00640  10/10 cells beat it

The pixel-i.i.d. control cancels exactly in this difference: both terms
use the same control_seed over the same images, so what is left is the
explainer scored against its own geometry.

========================================================================
P3 GATE   half-width < 10% of the clean-to-severity-5 change
========================================================================
imputation      explainer              family               change     half_w     ratio  verdict
black           grad_cam               defocus_blur       -0.11688    0.01897     0.162  FAIL
black           grad_cam               fog                -0.06316    0.01897     0.300  FAIL
black           grad_cam               gaussian_noise     +0.25987    0.02389     0.092  PASS
black           grad_cam_rolled        defocus_blur       -0.04353    0.01957     0.450  FAIL
black           grad_cam_rolled        fog                +0.01400    0.01957       n/a  no change
black           grad_cam_rolled        gaussian_noise     +0.28451    0.02520     0.089  PASS
black           integrated_gradients   defocus_blur       +0.01469    0.00793     0.540  FAIL
black           integrated_gradients   fog                +0.02997    0.00793     0.265  FAIL
black           integrated_gradients   gaussian_noise     +0.05729    0.00793     0.138  FAIL
black           integrated_gradients_rolled defocus_blur       +0.00992    0.00738     0.744  FAIL
black           integrated_gradients_rolled fog                +0.01962    0.00738     0.376  FAIL
black           integrated_gradients_rolled gaussian_noise     +0.06090    0.00739     0.121  FAIL
black           random                 defocus_blur       -0.00010    0.00012       n/a  no change
black           random                 fog                +0.00002    0.00012       n/a  no change
black           random                 gaussian_noise     +0.00006    0.00014       n/a  no change
black           random_lowres_16       defocus_blur       -0.00386    0.01211       n/a  no change
black           random_lowres_16       fog                +0.02449    0.01211     0.495  FAIL
black           random_lowres_16       gaussian_noise     +0.12609    0.01568     0.124  FAIL
black           random_lowres_2        defocus_blur       -0.04366    0.01936     0.443  FAIL
black           random_lowres_2        fog                -0.00916    0.01936       n/a  no change
black           random_lowres_2        gaussian_noise     +0.28455    0.02472     0.087  PASS
black           random_lowres_32       defocus_blur       -0.00905    0.00519     0.574  FAIL
black           random_lowres_32       fog                -0.00864    0.00519     0.601  FAIL
black           random_lowres_32       gaussian_noise     +0.02743    0.00595     0.217  FAIL
black           random_lowres_4        defocus_blur       -0.03825    0.01920     0.502  FAIL
black           random_lowres_4        fog                +0.00988    0.01920       n/a  no change
black           random_lowres_4        gaussian_noise     +0.27805    0.02373     0.085  PASS
black           random_lowres_8        defocus_blur       -0.01265    0.01712       n/a  no change
black           random_lowres_8        fog                +0.03062    0.01712     0.559  FAIL
black           random_lowres_8        gaussian_noise     +0.22625    0.02145     0.095  PASS
blur            grad_cam               defocus_blur       -0.07554    0.01203     0.159  FAIL
blur            grad_cam               fog                -0.01544    0.01203     0.779  FAIL
blur            grad_cam               gaussian_noise     +0.05738    0.01441     0.251  FAIL
blur            grad_cam_rolled        defocus_blur       -0.03006    0.01039     0.346  FAIL
blur            grad_cam_rolled        fog                -0.01967    0.01039     0.528  FAIL
blur            grad_cam_rolled        gaussian_noise     +0.05243    0.01372     0.262  FAIL
blur            integrated_gradients   defocus_blur       -0.00806    0.00850       n/a  no change
blur            integrated_gradients   fog                +0.04078    0.00936     0.229  FAIL
blur            integrated_gradients   gaussian_noise     -0.01367    0.01083     0.793  FAIL
blur            integrated_gradients_rolled defocus_blur       -0.01016    0.00730     0.718  FAIL
blur            integrated_gradients_rolled fog                +0.00507    0.00827       n/a  no change
blur            integrated_gradients_rolled gaussian_noise     +0.00237    0.01026       n/a  no change
blur            random                 defocus_blur       -0.00001    0.00010       n/a  no change
blur            random                 fog                -0.00002    0.00016       n/a  no change
blur            random                 gaussian_noise     -0.00004    0.00020       n/a  no change
blur            random_lowres_16       defocus_blur       -0.01108    0.00797     0.719  FAIL
blur            random_lowres_16       fog                -0.00608    0.00899       n/a  no change
blur            random_lowres_16       gaussian_noise     +0.01835    0.01180     0.643  FAIL
blur            random_lowres_2        defocus_blur       -0.04058    0.00983     0.242  FAIL
blur            random_lowres_2        fog                -0.02729    0.01050     0.385  FAIL
blur            random_lowres_2        gaussian_noise     +0.03980    0.01250     0.314  FAIL
blur            random_lowres_32       defocus_blur       -0.01456    0.00693     0.476  FAIL
blur            random_lowres_32       fog                -0.01550    0.00700     0.452  FAIL
blur            random_lowres_32       gaussian_noise     -0.00161    0.00975       n/a  no change
blur            random_lowres_4        defocus_blur       -0.03775    0.00991     0.262  FAIL
blur            random_lowres_4        fog                -0.02193    0.01035     0.472  FAIL
blur            random_lowres_4        gaussian_noise     +0.05033    0.01415     0.281  FAIL
blur            random_lowres_8        defocus_blur       -0.02018    0.00947     0.469  FAIL
blur            random_lowres_8        fog                -0.00630    0.01003       n/a  no change
blur            random_lowres_8        gaussian_noise     +0.04087    0.01294     0.317  FAIL
mean            grad_cam               defocus_blur       -0.05303    0.01716     0.324  FAIL
mean            grad_cam               fog                -0.02351    0.01716     0.730  FAIL
mean            grad_cam               gaussian_noise     +0.22499    0.02150     0.096  PASS
mean            grad_cam_rolled        defocus_blur       -0.00730    0.01874       n/a  no change
mean            grad_cam_rolled        fog                +0.01899    0.01874     0.986  FAIL
mean            grad_cam_rolled        gaussian_noise     +0.27281    0.02497     0.092  PASS
mean            integrated_gradients   defocus_blur       -0.02876    0.00699     0.243  FAIL
mean            integrated_gradients   fog                -0.01874    0.00718     0.383  FAIL
mean            integrated_gradients   gaussian_noise     +0.01846    0.00720     0.390  FAIL
mean            integrated_gradients_rolled defocus_blur       +0.00655    0.00801       n/a  no change
mean            integrated_gradients_rolled fog                +0.00791    0.00801       n/a  no change
mean            integrated_gradients_rolled gaussian_noise     +0.05079    0.00801     0.158  FAIL
mean            random                 defocus_blur       -0.00021    0.00016     0.759  FAIL
mean            random                 fog                -0.00011    0.00016       n/a  no change
mean            random                 gaussian_noise     -0.00011    0.00017       n/a  no change
mean            random_lowres_16       defocus_blur       +0.00952    0.01323       n/a  no change
mean            random_lowres_16       fog                +0.02889    0.01323     0.458  FAIL
mean            random_lowres_16       gaussian_noise     +0.14466    0.01535     0.106  FAIL
mean            random_lowres_2        defocus_blur       -0.01488    0.01750       n/a  no change
mean            random_lowres_2        fog                -0.00199    0.01750       n/a  no change
mean            random_lowres_2        gaussian_noise     +0.25696    0.02367     0.092  PASS
mean            random_lowres_32       defocus_blur       -0.00187    0.00630       n/a  no change
mean            random_lowres_32       fog                +0.00346    0.00630       n/a  no change
mean            random_lowres_32       gaussian_noise     +0.02063    0.00630     0.305  FAIL
mean            random_lowres_4        defocus_blur       -0.01744    0.01843       n/a  no change
mean            random_lowres_4        fog                +0.01300    0.01843       n/a  no change
mean            random_lowres_4        gaussian_noise     +0.26819    0.02353     0.088  PASS
mean            random_lowres_8        defocus_blur       +0.00263    0.01715       n/a  no change
mean            random_lowres_8        fog                +0.01851    0.01715     0.927  FAIL
mean            random_lowres_8        gaussian_noise     +0.23632    0.02195     0.093  PASS
uniform_noise   grad_cam               defocus_blur       +0.03018    0.00893     0.296  FAIL
uniform_noise   grad_cam               fog                +0.02266    0.00893     0.394  FAIL
uniform_noise   grad_cam               gaussian_noise     +0.07245    0.00893     0.123  FAIL
uniform_noise   grad_cam_rolled        defocus_blur       +0.05650    0.01358     0.240  FAIL
uniform_noise   grad_cam_rolled        fog                +0.05453    0.01358     0.249  FAIL
uniform_noise   grad_cam_rolled        gaussian_noise     +0.14823    0.01372     0.093  PASS
uniform_noise   integrated_gradients   defocus_blur       -0.01303    0.00519     0.398  FAIL
uniform_noise   integrated_gradients   fog                -0.01748    0.00517     0.296  FAIL
uniform_noise   integrated_gradients   gaussian_noise     +0.01746    0.00484     0.277  FAIL
uniform_noise   integrated_gradients_rolled defocus_blur       +0.01469    0.00593     0.403  FAIL
uniform_noise   integrated_gradients_rolled fog                +0.00869    0.00593     0.682  FAIL
uniform_noise   integrated_gradients_rolled gaussian_noise     +0.04308    0.00593     0.138  FAIL
uniform_noise   random                 defocus_blur       -0.00014    0.00011     0.832  FAIL
uniform_noise   random                 fog                -0.00012    0.00013       n/a  no change
uniform_noise   random                 gaussian_noise     -0.00002    0.00011       n/a  no change
uniform_noise   random_lowres_16       defocus_blur       +0.02339    0.00809     0.346  FAIL
uniform_noise   random_lowres_16       fog                +0.03352    0.00809     0.241  FAIL
uniform_noise   random_lowres_16       gaussian_noise     +0.06361    0.00809     0.127  FAIL
uniform_noise   random_lowres_2        defocus_blur       +0.05258    0.01281     0.244  FAIL
uniform_noise   random_lowres_2        fog                +0.05091    0.01281     0.252  FAIL
uniform_noise   random_lowres_2        gaussian_noise     +0.15386    0.01331     0.086  PASS
uniform_noise   random_lowres_32       defocus_blur       -0.00021    0.00361       n/a  no change
uniform_noise   random_lowres_32       fog                -0.00161    0.00361       n/a  no change
uniform_noise   random_lowres_32       gaussian_noise     +0.00138    0.00361       n/a  no change
uniform_noise   random_lowres_4        defocus_blur       +0.05240    0.01256     0.240  FAIL
uniform_noise   random_lowres_4        fog                +0.05734    0.01256     0.219  FAIL
uniform_noise   random_lowres_4        gaussian_noise     +0.14457    0.01277     0.088  PASS
uniform_noise   random_lowres_8        defocus_blur       +0.04608    0.01062     0.230  FAIL
uniform_noise   random_lowres_8        fog                +0.05257    0.01062     0.202  FAIL
uniform_noise   random_lowres_8        gaussian_noise     +0.11379    0.01091     0.096  PASS

GATE: 14/85 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================
config      /kaggle/working/shiftprofile/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
imputation  mean, blur
pairs       integrated_gradients vs integrated_gradients_rolled, grad_cam vs grad_cam_rolled
roll seed   0

==========================================================================
TEST 1   does rolling FURTHER help more?
         gain = faithfulness(rolled) - faithfulness(explainer)
         growing with displacement = reading (b); flat = reading (a)
==========================================================================
imputation      explainer              displ_px       gain    half_w      n
mean            integrated_gradients       5.73   -0.05582   0.00519   2040
mean            integrated_gradients       9.82   -0.05817   0.00546   2050
mean            integrated_gradients      12.56   -0.07612   0.00552   2100
mean            integrated_gradients      14.78   -0.08361   0.00565   2100
mean            integrated_gradients      17.90   -0.09381   0.00605   2030
mean            grad_cam                   5.73   -0.04357   0.00644   2040
mean            grad_cam                   9.82   -0.09261   0.00804   2050
mean            grad_cam                  12.56   -0.10935   0.00848   2100
mean            grad_cam                  14.78   -0.14116   0.00919   2100
mean            grad_cam                  17.90   -0.18904   0.00949   2030
blur            integrated_gradients       5.73   -0.05213   0.00643   2040
blur            integrated_gradients       9.82   -0.06957   0.00698   2050
blur            integrated_gradients      12.56   -0.06170   0.00651   2100
blur            integrated_gradients      14.78   -0.08415   0.00706   2100
blur            integrated_gradients      17.90   -0.08868   0.00778   2030
blur            grad_cam                   5.73   -0.03410   0.00515   2040
blur            grad_cam                   9.82   -0.07767   0.00697   2050
blur            grad_cam                  12.56   -0.08765   0.00808   2100
blur            grad_cam                  14.78   -0.12934   0.00840   2100
blur            grad_cam                  17.90   -0.15359   0.00932   2030

  blur            grad_cam               -0.03410 -> -0.15359  swing -0.11950 vs half-width 0.00932   SHRINKS with displacement
  blur            integrated_gradients   -0.05213 -> -0.08868  swing -0.03655 vs half-width 0.00778   SHRINKS with displacement
  mean            grad_cam               -0.04357 -> -0.18904  swing -0.14547 vs half-width 0.00949   SHRINKS with displacement
  mean            integrated_gradients   -0.05582 -> -0.09381  swing -0.03799 vs half-width 0.00605   SHRINKS with displacement

  [NOTE added 2026-10-15, not part of the run output. The four verdict labels
  immediately above are WRONG and are the opposite of the measurement. The
  comparison was made on signed swing, and `gain = faithfulness(rolled) -
  faithfulness(explainer)` is negative exactly when an explainer beats its own
  displaced copy, which is the working case. Read the numbers, not the label:
  every one of the four GROWS in magnitude with displacement -- blur/grad_cam
  0.03410 -> 0.15359 (4.5x), mean/grad_cam 0.04357 -> 0.18904 (4.3x),
  blur/IG 0.05213 -> 0.08868 (1.7x), mean/IG 0.05582 -> 0.09381 (1.7x) --
  against half-widths of 0.00605 to 0.00949. The label bug is fixed in
  scripts/mechanism_report.py at commit 77781e8; this log predates the fix. The
  corrected print is in Part 3, over numerically identical gains: all four rows
  there read `advantage GROWS with distance (beats its copy)`.]

==========================================================================
TEST 2   faithfulness against mask geometry, all arms on one table
         centre_distance: mask centroid's distance from frame centre
         spread: how far masked pixels sit from their own centroid
==========================================================================

  imputation: blur   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.117      +0.13321  
    random_lowres_2                  3.967  16.017      +0.04398  yes
    grad_cam_rolled                  6.150   9.262      +0.03673  
    random_lowres_4                  6.511  10.994      +0.04112  yes
    integrated_gradients             7.935   3.637      +0.07246  
    integrated_gradients_rolled     10.483   5.053      +0.00146  
    random_lowres_8                 10.897   6.229      +0.02673  yes
    random                          12.112   1.594      +0.00002  yes
    random_lowres_32                12.114   1.538      +0.01548  yes
    random_lowres_16                12.120   3.433      +0.00974  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.08923
      grad_cam_rolled              vs random_lowres_4      (spread 6.15 vs 6.51)  residual -0.00439
      integrated_gradients         vs random_lowres_4      (spread 7.93 vs 6.51)  residual +0.03134
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.48 vs 10.90)  residual -0.02527

  imputation: mean   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.117      -0.02421  
    random_lowres_2                  3.967  16.017      -0.15983  yes
    grad_cam_rolled                  6.150   9.262      -0.13944  
    random_lowres_4                  6.511  10.994      -0.12500  yes
    integrated_gradients             7.935   3.637      +0.02605  
    integrated_gradients_rolled     10.483   5.053      -0.04784  
    random_lowres_8                 10.897   6.229      -0.09769  yes
    random                          12.112   1.594      +0.00007  yes
    random_lowres_32                12.114   1.538      +0.00164  yes
    random_lowres_16                12.120   3.433      -0.07262  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.13562
      grad_cam_rolled              vs random_lowres_4      (spread 6.15 vs 6.51)  residual -0.01445
      integrated_gradients         vs random_lowres_4      (spread 7.93 vs 6.51)  residual +0.15104
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.48 vs 10.90)  residual +0.04985

==========================================================================
TEST 3   where in the removal curve the gap lives
         probability kept by the explainer's mask minus by a rolled one
==========================================================================

  blur / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.8117     0.8396   -0.0280
        0.10     0.7638     0.8190   -0.0552
        0.20     0.6833     0.7807   -0.0974
        0.30     0.6119     0.7421   -0.1302
        0.50     0.5097     0.6638   -0.1541
        0.70     0.4571     0.5768   -0.1197
        0.90     0.4351     0.4813   -0.0462
        1.00     0.4319     0.4319   +0.0000
    peak gap at fraction 0.50 (-0.1541)

  blur / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.7857     0.8430   -0.0573
        0.10     0.7534     0.8280   -0.0746
        0.20     0.7089     0.8014   -0.0926
        0.30     0.6755     0.7723   -0.0968
        0.50     0.6182     0.7097   -0.0915
        0.70     0.5607     0.6339   -0.0732
        0.90     0.4905     0.5294   -0.0389
        1.00     0.4319     0.4319   +0.0000
    peak gap at fraction 0.30 (-0.0968)

  mean / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.7038     0.7790   -0.0753
        0.10     0.5783     0.7181   -0.1398
        0.20     0.3953     0.6078   -0.2126
        0.30     0.2911     0.5062   -0.2151
        0.50     0.1912     0.3413   -0.1501
        0.70     0.1462     0.2211   -0.0749
        0.90     0.1227     0.1391   -0.0164
        1.00     0.1122     0.1122   +0.0000
    peak gap at fraction 0.30 (-0.2151)

  mean / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.5412     0.7040   -0.1628
        0.10     0.4216     0.6028   -0.1812
        0.20     0.2744     0.4399   -0.1656
        0.30     0.2070     0.3260   -0.1189
        0.50     0.1655     0.2199   -0.0544
        0.70     0.1510     0.1803   -0.0293
        0.90     0.1128     0.1196   -0.0068
        1.00     0.1122     0.1122   +0.0000
    peak gap at fraction 0.10 (-0.1812)

==========================================================================

---

## Part 3 — VALID: session 1, with the matched gate

**2026-10-15, commit `34e351a`. This is the run to cite.** Session 1 of the three
planned Kaggle sessions. Same config (`configs/pilot_validity.yaml`), same commit
lineage as Part 2, same model and seed -- `resnet18`, seed 0, 1000 images, 10
cells (clean + 3 families x 3 severities), 4 imputation schemes, 10 explainer
arms. 510 units, 0 failed, 8713.3s. Part 2 is superseded rather than retracted:
nothing in it is withdrawn, and Part 3 reproduces it exactly.

Three things make this run worth its own part: it reproduces Part 2 bit-for-bit,
it prints two tables Part 2 could not, and the geometry-matched gate -- the one
the metric decision actually reads -- fails completely.

### It reproduces Part 2 bit-for-bit

Direct string comparison against
`docs/runs/2026-10-15_curves-v3_pilot_validity.log`:

| block | result |
|---|---|
| CALIBRATION | byte-identical |
| FAITHFULNESS, summary and all 160 per-cell rows | byte-identical |
| GEOMETRY LADDER | byte-identical |
| GEOMETRY-MATCHED FAITHFULNESS | byte-identical |
| P3 GATE (RAW), 120 rows | identical in every field; the explainer column was widened from 23 to 28 characters to fit `integrated_gradients_rolled`, so the rows are not byte-identical |
| TEST 1 numeric table, 20 rows | byte-identical |
| TEST 2 and TEST 3 | byte-identical |
| training loss, all 11 printed epochs | identical, 1.5867 at epoch 1 through 0.0016 at epoch 50 |

This is not a formality. Neither session had a cache attached -- both logs print
`cache read  (none: /kaggle/input/shiftprofile-cache is not attached)` -- and
`train-v2` puts `epochs`, `batch_size` and `lr` into the training cache key, so in
each session the checkpoint was trained from scratch rather than loaded. Two
independent trainings, in two Kaggle sessions, produced weights that agree on
every number downstream of them: 160 faithfulness point estimates and their
bootstrap half-widths, to five decimals. That is an end-to-end determinism check
nobody designed, and it passed.

### Two tables Part 2 could not print

**`P3 GATE (GEOMETRY-MATCHED)` did not exist in the code when Part 2 ran.** It is
the substantive new content of this run and it is the subject of the next two
sections.

**TEST 1's verdict column is now correct.** All four rows read `advantage GROWS
with distance (beats its copy)`, where Part 2's log printed `SHRINKS with
displacement`. The gains underneath are numerically identical between the two
logs; only the label changed, at commit `77781e8`. Part 2's inline NOTE on those
stale labels stands.

### The matched gate fails completely: 0 of 29 resolvable pairs

The gate asks whether the bootstrap half-width is under 10% of the
clean-to-severity-5 change. On the geometry-matched metric no pair clears it:
**0 of 29 resolvable pairs pass**, against **14 of 85** on the raw metric. The six
closest, with the sample size each would need, since the half-width falls as
1/sqrt(n):

| imputation | pair | family | ratio | n needed |
|---|---|---|---|---|
| `black` | grad_cam vs random_lowres_4 | defocus_blur | 0.143 | 2,045 |
| `uniform_noise` | grad_cam vs random_lowres_4 | gaussian_noise | 0.152 | 2,310 |
| `black` | grad_cam vs random_lowres_4 | fog | 0.154 | 2,372 |
| `uniform_noise` | grad_cam vs grad_cam_rolled | gaussian_noise | 0.161 | 2,592 |
| `black` | grad_cam vs grad_cam_rolled | fog | 0.164 | 2,690 |
| `black` | grad_cam vs grad_cam_rolled | defocus_blur | 0.172 | 2,958 |

`n needed` is 1000 x (ratio/0.10)^2, taken from the ratio as the log prints it.
Recomputing from the unrounded change and half-width moves each entry by at most
11 images and changes none of the counts below.

Scaled across the whole set: **0 of 29 pass at n=2000**, 6 at n=3000, 7 at
n=5000, 19 at n=10000. The CIFAR-10 test set holds 10,000 images, so the last of
those is the entire test set.

**What the gate measures, and what it does not.** The gate reads the *change* in
matched faithfulness across severity. It does not read its *level*. The level is
resolved superbly: 11 of the 12 matched comparisons sit 7.3 to 10.4 half-widths
above zero, the exception being `black`/IG at -0.00398 and 0.6 half-widths, which
is unresolved rather than failed. So the corrected instrument can state that an
explainer beats a control of its own mask geometry, and cannot state whether that
advantage degrades under shift. Those are different quantities. The size of the
mask-geometry artefact, and the sign disagreement it produces between the raw and
matched metrics, both rest on the level.

### The degradation is unresolved, not absent

The point estimates are substantial. `mean change` is the average over the three
corruption families of the clean-to-severity-5 change in the matched metric, from
the P3 GATE (GEOMETRY-MATCHED) table below; `level` is the pair's matched
faithfulness from the GEOMETRY-MATCHED FAITHFULNESS table; `relative` is the
first as a fraction of the second. Over the same interval accuracy falls 36.3%,
0.8610 -> 0.5487.

| imputation | pair | mean change | level | relative |
|---|---|---|---|---|
| `black` | grad_cam vs grad_cam_rolled | -0.0584 | +0.1007 | -58.0% |
| `black` | grad_cam vs random_lowres_4 | -0.0566 | +0.0919 | -61.6% |
| `black` | IG vs IG_rolled | +0.0038 | -0.0040 | n/a, level unresolved |
| `blur` | grad_cam vs grad_cam_rolled | -0.0121 | +0.0965 | -12.5% |
| `blur` | grad_cam vs random_lowres_4 | -0.0081 | +0.0921 | -8.8% |
| `blur` | IG vs IG_rolled | +0.0073 | +0.0710 | +10.2% |
| `mean` | grad_cam vs grad_cam_rolled | -0.0454 | +0.1152 | -39.4% |
| `mean` | grad_cam vs random_lowres_4 | -0.0384 | +0.1008 | -38.1% |
| `mean` | IG vs IG_rolled | -0.0314 | +0.0739 | -42.5% |
| `uniform_noise` | grad_cam vs grad_cam_rolled | -0.0447 | +0.1208 | -37.0% |
| `uniform_noise` | grad_cam vs random_lowres_4 | -0.0430 | +0.1105 | -38.9% |
| `uniform_noise` | IG vs IG_rolled | -0.0265 | +0.0486 | -54.6% |

Under `black`, `mean` and `uniform_noise` the matched advantage falls by 37% to
62% of its own level. Five of those eight fall between 37% and 43%, within seven
points of accuracy's 36.3%; three -- `black` grad_cam against both of its
controls, and `uniform_noise` IG -- run from 55% to 62%.

`blur` is the exception and should be read as one. There the advantage barely
moves: falls of 8.8% and 12.5% for grad_cam, and a 10.2% rise for IG. `blur` is
also the only imputation scheme under which all ten arms read positive on the raw
metric, so it is the scheme where the imputation, and not the explainer, is doing
most of the work.

H3a predicts that faithfulness degrades *faster* than accuracy. All eight
interpretable non-`blur` pairs run in that direction and past it, five of them
only just -- within seven points of accuracy's 36.3% -- and three of them
clearly, at 54.6% to 61.6%, which is `black` against both its controls and
`uniform_noise` IG. The three `blur` pairs run the other way. None of the 29
changes is resolved at this n, which is precisely the statement that these
intervals cannot separate any of these rates from accuracy's. Record it as an
unresolved comparison with informative point estimates. It is not a null: a null
would assert the degradation is absent, and these numbers do not say that either.

### TEST 2: a displaced Grad-CAM map lands on the model-free line; a displaced IG map does not

TEST 2 orders every arm by mask spread and takes each real arm's residual against
the model-free arms of nearest spread. `grad_cam_rolled` -- a real Grad-CAM map,
displaced -- reads -0.00439 under `blur` and -0.01445 under `mean`, against a
comparator matched to within 0.36 of spread. It sits on the model-free line,
which is exactly what the geometry account predicts for a map whose geometry is
intact and whose aim has been destroyed. The correctly-aimed arms sit far above
it: grad_cam +0.08923 under `blur` and +0.13562 under `mean`, IG +0.03134 and
+0.15104. This is the geometry account's cleanest single confirmation in the run,
and it was not available before.

The fourth arm does not fit as cleanly. `integrated_gradients_rolled` reads
-0.02527 under `blur` and +0.04985 under `mean`, against a comparator matched to
within 0.41 of spread -- so under `mean` a displaced map scores nearly 0.05 above
the model-free line. The confirmation is clean for `grad_cam_rolled` and partial
for `integrated_gradients_rolled`.

### Operational note

CIFAR-10 was downloaded in-session again: `170M/170M [51:15<00:00, 55.4kB/s]`,
against `[22:42<00:00, 125kB/s]` in the Part 2 run. So roughly 51 minutes of the
2.42-hour session was a download that an attached dataset would have skipped, and
the 8713.3s total is not comparable with Part 2's 7325.1s as a compute figure.

### Scope

Single seed throughout: `resnet18`, seed 0, 1000 images, one dataset. Everything
above is one model and one seed, and nothing in it generalises past that until
the remaining two sessions are in.

Raw output below, copied unedited from
`docs/runs/2026-10-15b_curves-v3_with_matched_gate.log`, from the run summary
onward.

============================================================
Completed 510 cells, skipped 0 cached, failed 0 cells
Elapsed time: 8713.3s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean, blur, uniform_noise, black
explainers  integrated_gradients, grad_cam, random, integrated_gradients_rolled, grad_cam_rolled, random_lowres_2, random_lowres_4, random_lowres_8, random_lowres_16, random_lowres_32

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8610    0.2166    0.0516    0.0764    0.0315
   1     0.8330    0.2522    0.0619    0.0790    0.0430
   3     0.7247    0.4160    0.1318    0.1495    0.1114
   5     0.5487    0.6810    0.2514    0.2650    0.2702

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================

  imputation: black
    grad_cam               mean -0.04735   2/10 cells above 0  
    grad_cam_rolled        mean -0.14802   1/10 cells above 0  
    integrated_gradients   mean -0.04214   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.03817   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.02958   2/10 cells above 0  
    random_lowres_2        mean -0.19932   1/10 cells above 0  
    random_lowres_32       mean +0.00531   3/10 cells above 0  
    random_lowres_4        mean -0.13929   1/10 cells above 0  
    random_lowres_8        mean -0.07275   2/10 cells above 0  

  imputation: blur
    grad_cam               mean +0.13321   10/10 cells above 0  
    grad_cam_rolled        mean +0.03673   9/10 cells above 0  
    integrated_gradients   mean +0.07246   10/10 cells above 0  
    integrated_gradients_rolled mean +0.00146   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean +0.00974   5/10 cells above 0  
    random_lowres_2        mean +0.04398   9/10 cells above 0  
    random_lowres_32       mean +0.01548   9/10 cells above 0  
    random_lowres_4        mean +0.04112   9/10 cells above 0  
    random_lowres_8        mean +0.02673   9/10 cells above 0  

  imputation: mean
    grad_cam               mean -0.02421   2/10 cells above 0  
    grad_cam_rolled        mean -0.13944   1/10 cells above 0  
    integrated_gradients   mean +0.02605   9/10 cells above 0  
    integrated_gradients_rolled mean -0.04784   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00007   3/10 cells above 0  
    random_lowres_16       mean -0.07262   1/10 cells above 0  
    random_lowres_2        mean -0.15983   1/10 cells above 0  
    random_lowres_32       mean +0.00164   2/10 cells above 0  
    random_lowres_4        mean -0.12500   1/10 cells above 0  
    random_lowres_8        mean -0.09769   2/10 cells above 0  

  imputation: uniform_noise
    grad_cam               mean -0.04382   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.16466   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients   mean +0.00203   3/10 cells above 0  
    integrated_gradients_rolled mean -0.04652   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.06960   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_2        mean -0.18550   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean -0.00027   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.15432   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_8        mean -0.11924   0/10 cells above 0  NO CELL BEATS RANDOM

imputation      explainer               sev   faithfulness   half_width
black           grad_cam                  0       -0.07593      0.01897
black           grad_cam                  1       -0.06058      0.01979
black           grad_cam                  3       -0.02262      0.02042
black           grad_cam                  5       -0.04932      0.01969
black           grad_cam_rolled           0       -0.20265      0.01957
black           grad_cam_rolled           1       -0.18291      0.02040
black           grad_cam_rolled           3       -0.12530      0.02105
black           grad_cam_rolled           5       -0.11765      0.02001
black           integrated_gradients      0       -0.06139      0.00793
black           integrated_gradients      1       -0.05478      0.00796
black           integrated_gradients      3       -0.03783      0.00750
black           integrated_gradients      5       -0.02741      0.00651
black           integrated_gradients_rolled    0       -0.05671      0.00738
black           integrated_gradients_rolled    1       -0.04735      0.00761
black           integrated_gradients_rolled    3       -0.03441      0.00732
black           integrated_gradients_rolled    5       -0.02656      0.00676
black           random                    0       +0.00002      0.00012
black           random                    1       +0.00004      0.00012
black           random                    3       +0.00001      0.00012
black           random                    5       +0.00001      0.00011
black           random_lowres_16          0       -0.06017      0.01211
black           random_lowres_16          1       -0.04876      0.01250
black           random_lowres_16          3       -0.01853      0.01308
black           random_lowres_16          5       -0.01126      0.01179
black           random_lowres_2           0       -0.24898      0.01936
black           random_lowres_2           1       -0.23352      0.02001
black           random_lowres_2           3       -0.17614      0.02126
black           random_lowres_2           5       -0.17173      0.02023
black           random_lowres_32          0       +0.00216      0.00519
black           random_lowres_32          1       +0.00406      0.00502
black           random_lowres_32          3       +0.00752      0.00491
black           random_lowres_32          5       +0.00541      0.00451
black           random_lowres_4           0       -0.19355      0.01920
black           random_lowres_4           1       -0.17360      0.01963
black           random_lowres_4           3       -0.11586      0.02037
black           random_lowres_4           5       -0.11032      0.01923
black           random_lowres_8           0       -0.12366      0.01712
black           random_lowres_8           1       -0.10552      0.01796
black           random_lowres_8           3       -0.05350      0.01886
black           random_lowres_8           5       -0.04225      0.01675
blur            grad_cam                  0       +0.13101      0.01203
blur            grad_cam                  1       +0.13677      0.01263
blur            grad_cam                  3       +0.14379      0.01271
blur            grad_cam                  5       +0.11980      0.01226
blur            grad_cam_rolled           0       +0.03306      0.01039
blur            grad_cam_rolled           1       +0.03763      0.01075
blur            grad_cam_rolled           3       +0.03982      0.01185
blur            grad_cam_rolled           5       +0.03397      0.01113
blur            integrated_gradients      0       +0.06346      0.00850
blur            integrated_gradients      1       +0.06672      0.00910
blur            integrated_gradients      3       +0.08385      0.00942
blur            integrated_gradients      5       +0.06982      0.00946
blur            integrated_gradients_rolled    0       +0.00019      0.00730
blur            integrated_gradients_rolled    1       +0.00188      0.00792
blur            integrated_gradients_rolled    3       +0.00363      0.00876
blur            integrated_gradients_rolled    5       -0.00072      0.00834
blur            random                    0       +0.00004      0.00010
blur            random                    1       +0.00006      0.00011
blur            random                    3       -0.00004      0.00012
blur            random                    5       +0.00001      0.00015
blur            random_lowres_16          0       +0.00992      0.00797
blur            random_lowres_16          1       +0.00945      0.00848
blur            random_lowres_16          3       +0.00939      0.00914
blur            random_lowres_16          5       +0.01032      0.00947
blur            random_lowres_2           0       +0.04713      0.00983
blur            random_lowres_2           1       +0.04785      0.01054
blur            random_lowres_2           3       +0.04528      0.01112
blur            random_lowres_2           5       +0.03777      0.01078
blur            random_lowres_32          0       +0.02070      0.00693
blur            random_lowres_32          1       +0.02112      0.00714
blur            random_lowres_32          3       +0.01343      0.00715
blur            random_lowres_32          5       +0.01015      0.00754
blur            random_lowres_4           0       +0.04006      0.00991
blur            random_lowres_4           1       +0.04183      0.01114
blur            random_lowres_4           3       +0.04494      0.01120
blur            random_lowres_4           5       +0.03694      0.01123
blur            random_lowres_8           0       +0.02293      0.00947
blur            random_lowres_8           1       +0.02452      0.01017
blur            random_lowres_8           3       +0.02922      0.01052
blur            random_lowres_8           5       +0.02772      0.01055
mean            grad_cam                  0       -0.05815      0.01716
mean            grad_cam                  1       -0.04436      0.01726
mean            grad_cam                  3       -0.00829      0.01779
mean            grad_cam                  5       -0.00867      0.01677
mean            grad_cam_rolled           0       -0.19263      0.01874
mean            grad_cam_rolled           1       -0.17750      0.01932
mean            grad_cam_rolled           3       -0.12531      0.02006
mean            grad_cam_rolled           5       -0.09780      0.01897
mean            integrated_gradients      0       +0.02810      0.00699
mean            integrated_gradients      1       +0.03020      0.00730
mean            integrated_gradients      3       +0.02883      0.00715
mean            integrated_gradients      5       +0.01842      0.00703
mean            integrated_gradients_rolled    0       -0.06057      0.00801
mean            integrated_gradients_rolled    1       -0.05669      0.00802
mean            integrated_gradients_rolled    3       -0.04377      0.00813
mean            integrated_gradients_rolled    5       -0.03882      0.00758
mean            random                    0       +0.00017      0.00016
mean            random                    1       +0.00014      0.00015
mean            random                    3       +0.00003      0.00015
mean            random                    5       +0.00002      0.00015
mean            random_lowres_16          0       -0.10768      0.01323
mean            random_lowres_16          1       -0.09647      0.01313
mean            random_lowres_16          3       -0.06305      0.01368
mean            random_lowres_16          5       -0.04666      0.01261
mean            random_lowres_2           0       -0.20590      0.01750
mean            random_lowres_2           1       -0.19426      0.01822
mean            random_lowres_2           3       -0.14401      0.01861
mean            random_lowres_2           5       -0.12587      0.01796
mean            random_lowres_32          0       -0.00309      0.00630
mean            random_lowres_32          1       -0.00080      0.00619
mean            random_lowres_32          3       +0.00300      0.00636
mean            random_lowres_32          5       +0.00431      0.00602
mean            random_lowres_4           0       -0.17507      0.01843
mean            random_lowres_4           1       -0.16162      0.01853
mean            random_lowres_4           3       -0.10952      0.01965
mean            random_lowres_4           5       -0.08715      0.01821
mean            random_lowres_8           0       -0.14582      0.01715
mean            random_lowres_8           1       -0.13206      0.01735
mean            random_lowres_8           3       -0.08495      0.01773
mean            random_lowres_8           5       -0.06000      0.01688
uniform_noise   grad_cam                  0       -0.06842      0.00893
uniform_noise   grad_cam                  1       -0.05924      0.00897
uniform_noise   grad_cam                  3       -0.03737      0.00809
uniform_noise   grad_cam                  5       -0.02665      0.00789
uniform_noise   grad_cam_rolled           0       -0.21145      0.01358
uniform_noise   grad_cam_rolled           1       -0.19869      0.01366
uniform_noise   grad_cam_rolled           3       -0.15465      0.01314
uniform_noise   grad_cam_rolled           5       -0.12502      0.01230
uniform_noise   integrated_gradients      0       +0.00260      0.00484
uniform_noise   integrated_gradients      1       +0.00437      0.00504
uniform_noise   integrated_gradients      3       +0.00328      0.00485
uniform_noise   integrated_gradients      5       -0.00175      0.00487
uniform_noise   integrated_gradients_rolled    0       -0.06007      0.00593
uniform_noise   integrated_gradients_rolled    1       -0.05529      0.00605
uniform_noise   integrated_gradients_rolled    3       -0.04184      0.00595
uniform_noise   integrated_gradients_rolled    5       -0.03791      0.00550
uniform_noise   random                    0       +0.00008      0.00011
uniform_noise   random                    1       +0.00009      0.00012
uniform_noise   random                    3       -0.00003      0.00012
uniform_noise   random                    5       -0.00001      0.00011
uniform_noise   random_lowres_16          0       -0.09149      0.00809
uniform_noise   random_lowres_16          1       -0.08548      0.00808
uniform_noise   random_lowres_16          3       -0.06471      0.00788
uniform_noise   random_lowres_16          5       -0.05132      0.00708
uniform_noise   random_lowres_2           0       -0.23212      0.01281
uniform_noise   random_lowres_2           1       -0.21970      0.01313
uniform_noise   random_lowres_2           3       -0.17493      0.01289
uniform_noise   random_lowres_2           5       -0.14633      0.01187
uniform_noise   random_lowres_32          0       -0.00065      0.00361
uniform_noise   random_lowres_32          1       -0.00047      0.00347
uniform_noise   random_lowres_32          3       +0.00057      0.00346
uniform_noise   random_lowres_32          5       -0.00080      0.00325
uniform_noise   random_lowres_4           0       -0.19989      0.01256
uniform_noise   random_lowres_4           1       -0.18862      0.01252
uniform_noise   random_lowres_4           3       -0.14404      0.01266
uniform_noise   random_lowres_4           5       -0.11512      0.01150
uniform_noise   random_lowres_8           0       -0.15836      0.01062
uniform_noise   random_lowres_8           1       -0.14728      0.01053
uniform_noise   random_lowres_8           3       -0.10984      0.01060
uniform_noise   random_lowres_8           5       -0.08755      0.00966

========================================================================
GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid
                  lattice, upsampled as Grad-CAM is. No model input.
========================================================================
imputation       grid   faithfulness   half_width
black               2       -0.19932      0.02039
black               4       -0.13929      0.01969
black               8       -0.07275      0.01778
black              16       -0.02958      0.01242
black              32       +0.00531      0.00485
blur                2       +0.04398      0.01071
blur                4       +0.04112      0.01106
blur                8       +0.02673      0.01032
blur               16       +0.00974      0.00892
blur               32       +0.01548      0.00724
mean                2       -0.15983      0.01819
mean                4       -0.12500      0.01876
mean                8       -0.09769      0.01730
mean               16       -0.07262      0.01315
mean               32       +0.00164      0.00620
uniform_noise       2       -0.18550      0.01265
uniform_noise       4       -0.15432      0.01226
uniform_noise       8       -0.11924      0.01030
uniform_noise      16       -0.06960      0.00772
uniform_noise      32       -0.00027      0.00341

Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on this
curve at grid=4, its score is its resolution and not its content.

========================================================================
GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the
                                same mask geometry; >0 beats it
========================================================================
imputation      explainer                vs                   kind          matched     half_w  verdict
black           grad_cam                 grad_cam_rolled      alignment    +0.10068    0.01169  10/10 cells beat it
black           grad_cam                 random_lowres_4      geometry     +0.09194    0.01088  10/10 cells beat it
black           integrated_gradients     integrated_gradients_rolled alignment    -0.00398    0.00638  1/10 cells beat it
blur            grad_cam                 grad_cam_rolled      alignment    +0.09648    0.01107  10/10 cells beat it
blur            grad_cam                 random_lowres_4      geometry     +0.09209    0.01046  10/10 cells beat it
blur            integrated_gradients     integrated_gradients_rolled alignment    +0.07100    0.00973  10/10 cells beat it
mean            grad_cam                 grad_cam_rolled      alignment    +0.11523    0.01233  10/10 cells beat it
mean            grad_cam                 random_lowres_4      geometry     +0.10078    0.01160  10/10 cells beat it
mean            integrated_gradients     integrated_gradients_rolled alignment    +0.07389    0.00810  10/10 cells beat it
uniform_noise   grad_cam                 grad_cam_rolled      alignment    +0.12083    0.01160  10/10 cells beat it
uniform_noise   grad_cam                 random_lowres_4      geometry     +0.11050    0.01085  10/10 cells beat it
uniform_noise   integrated_gradients     integrated_gradients_rolled alignment    +0.04855    0.00640  10/10 cells beat it

The pixel-i.i.d. control cancels exactly in this difference: both terms
use the same control_seed over the same images, so what is left is the
explainer scored against its own geometry.

========================================================================
P3 GATE (RAW)   half-width < 10% of the clean-to-severity-5 change
                on `faithfulness`: the explainer against a pixel-iid
                control. The validity run showed that quantity tracks
                mask geometry -- a model-free blob scores well below
                zero on it -- so this table is reported, not read.
========================================================================
imputation      explainer                   family               change     half_w     ratio  verdict
black           grad_cam                    defocus_blur       -0.11688    0.01897     0.162  FAIL
black           grad_cam                    fog                -0.06316    0.01897     0.300  FAIL
black           grad_cam                    gaussian_noise     +0.25987    0.02389     0.092  PASS
black           grad_cam_rolled             defocus_blur       -0.04353    0.01957     0.450  FAIL
black           grad_cam_rolled             fog                +0.01400    0.01957       n/a  no change
black           grad_cam_rolled             gaussian_noise     +0.28451    0.02520     0.089  PASS
black           integrated_gradients        defocus_blur       +0.01469    0.00793     0.540  FAIL
black           integrated_gradients        fog                +0.02997    0.00793     0.265  FAIL
black           integrated_gradients        gaussian_noise     +0.05729    0.00793     0.138  FAIL
black           integrated_gradients_rolled defocus_blur       +0.00992    0.00738     0.744  FAIL
black           integrated_gradients_rolled fog                +0.01962    0.00738     0.376  FAIL
black           integrated_gradients_rolled gaussian_noise     +0.06090    0.00739     0.121  FAIL
black           random                      defocus_blur       -0.00010    0.00012       n/a  no change
black           random                      fog                +0.00002    0.00012       n/a  no change
black           random                      gaussian_noise     +0.00006    0.00014       n/a  no change
black           random_lowres_16            defocus_blur       -0.00386    0.01211       n/a  no change
black           random_lowres_16            fog                +0.02449    0.01211     0.495  FAIL
black           random_lowres_16            gaussian_noise     +0.12609    0.01568     0.124  FAIL
black           random_lowres_2             defocus_blur       -0.04366    0.01936     0.443  FAIL
black           random_lowres_2             fog                -0.00916    0.01936       n/a  no change
black           random_lowres_2             gaussian_noise     +0.28455    0.02472     0.087  PASS
black           random_lowres_32            defocus_blur       -0.00905    0.00519     0.574  FAIL
black           random_lowres_32            fog                -0.00864    0.00519     0.601  FAIL
black           random_lowres_32            gaussian_noise     +0.02743    0.00595     0.217  FAIL
black           random_lowres_4             defocus_blur       -0.03825    0.01920     0.502  FAIL
black           random_lowres_4             fog                +0.00988    0.01920       n/a  no change
black           random_lowres_4             gaussian_noise     +0.27805    0.02373     0.085  PASS
black           random_lowres_8             defocus_blur       -0.01265    0.01712       n/a  no change
black           random_lowres_8             fog                +0.03062    0.01712     0.559  FAIL
black           random_lowres_8             gaussian_noise     +0.22625    0.02145     0.095  PASS
blur            grad_cam                    defocus_blur       -0.07554    0.01203     0.159  FAIL
blur            grad_cam                    fog                -0.01544    0.01203     0.779  FAIL
blur            grad_cam                    gaussian_noise     +0.05738    0.01441     0.251  FAIL
blur            grad_cam_rolled             defocus_blur       -0.03006    0.01039     0.346  FAIL
blur            grad_cam_rolled             fog                -0.01967    0.01039     0.528  FAIL
blur            grad_cam_rolled             gaussian_noise     +0.05243    0.01372     0.262  FAIL
blur            integrated_gradients        defocus_blur       -0.00806    0.00850       n/a  no change
blur            integrated_gradients        fog                +0.04078    0.00936     0.229  FAIL
blur            integrated_gradients        gaussian_noise     -0.01367    0.01083     0.793  FAIL
blur            integrated_gradients_rolled defocus_blur       -0.01016    0.00730     0.718  FAIL
blur            integrated_gradients_rolled fog                +0.00507    0.00827       n/a  no change
blur            integrated_gradients_rolled gaussian_noise     +0.00237    0.01026       n/a  no change
blur            random                      defocus_blur       -0.00001    0.00010       n/a  no change
blur            random                      fog                -0.00002    0.00016       n/a  no change
blur            random                      gaussian_noise     -0.00004    0.00020       n/a  no change
blur            random_lowres_16            defocus_blur       -0.01108    0.00797     0.719  FAIL
blur            random_lowres_16            fog                -0.00608    0.00899       n/a  no change
blur            random_lowres_16            gaussian_noise     +0.01835    0.01180     0.643  FAIL
blur            random_lowres_2             defocus_blur       -0.04058    0.00983     0.242  FAIL
blur            random_lowres_2             fog                -0.02729    0.01050     0.385  FAIL
blur            random_lowres_2             gaussian_noise     +0.03980    0.01250     0.314  FAIL
blur            random_lowres_32            defocus_blur       -0.01456    0.00693     0.476  FAIL
blur            random_lowres_32            fog                -0.01550    0.00700     0.452  FAIL
blur            random_lowres_32            gaussian_noise     -0.00161    0.00975       n/a  no change
blur            random_lowres_4             defocus_blur       -0.03775    0.00991     0.262  FAIL
blur            random_lowres_4             fog                -0.02193    0.01035     0.472  FAIL
blur            random_lowres_4             gaussian_noise     +0.05033    0.01415     0.281  FAIL
blur            random_lowres_8             defocus_blur       -0.02018    0.00947     0.469  FAIL
blur            random_lowres_8             fog                -0.00630    0.01003       n/a  no change
blur            random_lowres_8             gaussian_noise     +0.04087    0.01294     0.317  FAIL
mean            grad_cam                    defocus_blur       -0.05303    0.01716     0.324  FAIL
mean            grad_cam                    fog                -0.02351    0.01716     0.730  FAIL
mean            grad_cam                    gaussian_noise     +0.22499    0.02150     0.096  PASS
mean            grad_cam_rolled             defocus_blur       -0.00730    0.01874       n/a  no change
mean            grad_cam_rolled             fog                +0.01899    0.01874     0.986  FAIL
mean            grad_cam_rolled             gaussian_noise     +0.27281    0.02497     0.092  PASS
mean            integrated_gradients        defocus_blur       -0.02876    0.00699     0.243  FAIL
mean            integrated_gradients        fog                -0.01874    0.00718     0.383  FAIL
mean            integrated_gradients        gaussian_noise     +0.01846    0.00720     0.390  FAIL
mean            integrated_gradients_rolled defocus_blur       +0.00655    0.00801       n/a  no change
mean            integrated_gradients_rolled fog                +0.00791    0.00801       n/a  no change
mean            integrated_gradients_rolled gaussian_noise     +0.05079    0.00801     0.158  FAIL
mean            random                      defocus_blur       -0.00021    0.00016     0.759  FAIL
mean            random                      fog                -0.00011    0.00016       n/a  no change
mean            random                      gaussian_noise     -0.00011    0.00017       n/a  no change
mean            random_lowres_16            defocus_blur       +0.00952    0.01323       n/a  no change
mean            random_lowres_16            fog                +0.02889    0.01323     0.458  FAIL
mean            random_lowres_16            gaussian_noise     +0.14466    0.01535     0.106  FAIL
mean            random_lowres_2             defocus_blur       -0.01488    0.01750       n/a  no change
mean            random_lowres_2             fog                -0.00199    0.01750       n/a  no change
mean            random_lowres_2             gaussian_noise     +0.25696    0.02367     0.092  PASS
mean            random_lowres_32            defocus_blur       -0.00187    0.00630       n/a  no change
mean            random_lowres_32            fog                +0.00346    0.00630       n/a  no change
mean            random_lowres_32            gaussian_noise     +0.02063    0.00630     0.305  FAIL
mean            random_lowres_4             defocus_blur       -0.01744    0.01843       n/a  no change
mean            random_lowres_4             fog                +0.01300    0.01843       n/a  no change
mean            random_lowres_4             gaussian_noise     +0.26819    0.02353     0.088  PASS
mean            random_lowres_8             defocus_blur       +0.00263    0.01715       n/a  no change
mean            random_lowres_8             fog                +0.01851    0.01715     0.927  FAIL
mean            random_lowres_8             gaussian_noise     +0.23632    0.02195     0.093  PASS
uniform_noise   grad_cam                    defocus_blur       +0.03018    0.00893     0.296  FAIL
uniform_noise   grad_cam                    fog                +0.02266    0.00893     0.394  FAIL
uniform_noise   grad_cam                    gaussian_noise     +0.07245    0.00893     0.123  FAIL
uniform_noise   grad_cam_rolled             defocus_blur       +0.05650    0.01358     0.240  FAIL
uniform_noise   grad_cam_rolled             fog                +0.05453    0.01358     0.249  FAIL
uniform_noise   grad_cam_rolled             gaussian_noise     +0.14823    0.01372     0.093  PASS
uniform_noise   integrated_gradients        defocus_blur       -0.01303    0.00519     0.398  FAIL
uniform_noise   integrated_gradients        fog                -0.01748    0.00517     0.296  FAIL
uniform_noise   integrated_gradients        gaussian_noise     +0.01746    0.00484     0.277  FAIL
uniform_noise   integrated_gradients_rolled defocus_blur       +0.01469    0.00593     0.403  FAIL
uniform_noise   integrated_gradients_rolled fog                +0.00869    0.00593     0.682  FAIL
uniform_noise   integrated_gradients_rolled gaussian_noise     +0.04308    0.00593     0.138  FAIL
uniform_noise   random                      defocus_blur       -0.00014    0.00011     0.832  FAIL
uniform_noise   random                      fog                -0.00012    0.00013       n/a  no change
uniform_noise   random                      gaussian_noise     -0.00002    0.00011       n/a  no change
uniform_noise   random_lowres_16            defocus_blur       +0.02339    0.00809     0.346  FAIL
uniform_noise   random_lowres_16            fog                +0.03352    0.00809     0.241  FAIL
uniform_noise   random_lowres_16            gaussian_noise     +0.06361    0.00809     0.127  FAIL
uniform_noise   random_lowres_2             defocus_blur       +0.05258    0.01281     0.244  FAIL
uniform_noise   random_lowres_2             fog                +0.05091    0.01281     0.252  FAIL
uniform_noise   random_lowres_2             gaussian_noise     +0.15386    0.01331     0.086  PASS
uniform_noise   random_lowres_32            defocus_blur       -0.00021    0.00361       n/a  no change
uniform_noise   random_lowres_32            fog                -0.00161    0.00361       n/a  no change
uniform_noise   random_lowres_32            gaussian_noise     +0.00138    0.00361       n/a  no change
uniform_noise   random_lowres_4             defocus_blur       +0.05240    0.01256     0.240  FAIL
uniform_noise   random_lowres_4             fog                +0.05734    0.01256     0.219  FAIL
uniform_noise   random_lowres_4             gaussian_noise     +0.14457    0.01277     0.088  PASS
uniform_noise   random_lowres_8             defocus_blur       +0.04608    0.01062     0.230  FAIL
uniform_noise   random_lowres_8             fog                +0.05257    0.01062     0.202  FAIL
uniform_noise   random_lowres_8             gaussian_noise     +0.11379    0.01091     0.096  PASS

GATE (RAW): 14/85 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================

========================================================================
P3 GATE (GEOMETRY-MATCHED)
   half-width < 10% of the clean-to-severity-5 change in `matched`:
   the explainer minus a control of its OWN mask geometry.
   THIS is the gate the 1000-vs-2000 decision reads. The raw
   gate above gates a quantity the validity run showed
   confounded, so neither its passes nor its failures bear on
   that decision.
========================================================================
imputation      explainer vs reference                              family               change     half_w     ratio  verdict
black           grad_cam vs grad_cam_rolled                         defocus_blur       -0.07335    0.01263     0.172  FAIL
black           grad_cam vs grad_cam_rolled                         fog                -0.07716    0.01263     0.164  FAIL
black           grad_cam vs grad_cam_rolled                         gaussian_noise     -0.02464    0.01263     0.512  FAIL
black           grad_cam vs random_lowres_4                         defocus_blur       -0.07864    0.01126     0.143  FAIL
black           grad_cam vs random_lowres_4                         fog                -0.07304    0.01125     0.154  FAIL
black           grad_cam vs random_lowres_4                         gaussian_noise     -0.01818    0.01125     0.619  FAIL
black           integrated_gradients vs integrated_gradients_rolled defocus_blur       +0.00477    0.00708       n/a  no change
black           integrated_gradients vs integrated_gradients_rolled fog                +0.01035    0.00708     0.684  FAIL
black           integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.00361    0.00708       n/a  no change
blur            grad_cam vs grad_cam_rolled                         defocus_blur       -0.04548    0.01105     0.243  FAIL
blur            grad_cam vs grad_cam_rolled                         fog                +0.00423    0.01105       n/a  no change
blur            grad_cam vs grad_cam_rolled                         gaussian_noise     +0.00494    0.01206       n/a  no change
blur            grad_cam vs random_lowres_4                         defocus_blur       -0.03779    0.01033     0.273  FAIL
blur            grad_cam vs random_lowres_4                         fog                +0.00648    0.01067       n/a  no change
blur            grad_cam vs random_lowres_4                         gaussian_noise     +0.00705    0.01199       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled defocus_blur       +0.00210    0.00919       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled fog                +0.03571    0.00996     0.279  FAIL
blur            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.01603    0.01133     0.707  FAIL
mean            grad_cam vs grad_cam_rolled                         defocus_blur       -0.04573    0.01224     0.268  FAIL
mean            grad_cam vs grad_cam_rolled                         fog                -0.04250    0.01224     0.288  FAIL
mean            grad_cam vs grad_cam_rolled                         gaussian_noise     -0.04782    0.01224     0.256  FAIL
mean            grad_cam vs random_lowres_4                         defocus_blur       -0.03558    0.01211     0.340  FAIL
mean            grad_cam vs random_lowres_4                         fog                -0.03651    0.01211     0.332  FAIL
mean            grad_cam vs random_lowres_4                         gaussian_noise     -0.04320    0.01211     0.280  FAIL
mean            integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.03531    0.00786     0.222  FAIL
mean            integrated_gradients vs integrated_gradients_rolled fog                -0.02666    0.00805     0.302  FAIL
mean            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.03234    0.00851     0.263  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         defocus_blur       -0.02632    0.01218     0.463  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         fog                -0.03187    0.01218     0.382  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         gaussian_noise     -0.07578    0.01218     0.161  FAIL
uniform_noise   grad_cam vs random_lowres_4                         defocus_blur       -0.02222    0.01098     0.494  FAIL
uniform_noise   grad_cam vs random_lowres_4                         fog                -0.03468    0.01098     0.317  FAIL
uniform_noise   grad_cam vs random_lowres_4                         gaussian_noise     -0.07212    0.01098     0.152  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.02773    0.00675     0.243  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled fog                -0.02617    0.00675     0.258  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.02563    0.00675     0.263  FAIL

GATE (GEOMETRY-MATCHED): 0/29 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
imputation  mean, blur
pairs       integrated_gradients vs integrated_gradients_rolled, grad_cam vs grad_cam_rolled
roll seed   0

==========================================================================
TEST 1   does displacing the map FURTHER cost more?
         gain = faithfulness(rolled) - faithfulness(explainer)
         gain < 0 means the explainer beats its own displaced copy;
         |gain| growing with distance means its aim is spatially
         specific, not merely a differently shaped mask
==========================================================================
imputation      explainer              displ_px       gain    half_w      n
mean            integrated_gradients       5.73   -0.05582   0.00519   2040
mean            integrated_gradients       9.82   -0.05817   0.00546   2050
mean            integrated_gradients      12.56   -0.07612   0.00552   2100
mean            integrated_gradients      14.78   -0.08361   0.00565   2100
mean            integrated_gradients      17.90   -0.09381   0.00605   2030
mean            grad_cam                   5.73   -0.04357   0.00644   2040
mean            grad_cam                   9.82   -0.09261   0.00804   2050
mean            grad_cam                  12.56   -0.10935   0.00848   2100
mean            grad_cam                  14.78   -0.14116   0.00919   2100
mean            grad_cam                  17.90   -0.18904   0.00949   2030
blur            integrated_gradients       5.73   -0.05213   0.00643   2040
blur            integrated_gradients       9.82   -0.06957   0.00698   2050
blur            integrated_gradients      12.56   -0.06170   0.00651   2100
blur            integrated_gradients      14.78   -0.08415   0.00706   2100
blur            integrated_gradients      17.90   -0.08868   0.00778   2030
blur            grad_cam                   5.73   -0.03410   0.00515   2040
blur            grad_cam                   9.82   -0.07767   0.00697   2050
blur            grad_cam                  12.56   -0.08765   0.00808   2100
blur            grad_cam                  14.78   -0.12934   0.00840   2100
blur            grad_cam                  17.90   -0.15359   0.00932   2030

  blur            grad_cam               -0.03410 -> -0.15359  |swing| +0.11950 vs half-width 0.00932   advantage GROWS with distance (beats its copy)
  blur            integrated_gradients   -0.05213 -> -0.08868  |swing| +0.03655 vs half-width 0.00778   advantage GROWS with distance (beats its copy)
  mean            grad_cam               -0.04357 -> -0.18904  |swing| +0.14547 vs half-width 0.00949   advantage GROWS with distance (beats its copy)
  mean            integrated_gradients   -0.05582 -> -0.09381  |swing| +0.03799 vs half-width 0.00605   advantage GROWS with distance (beats its copy)

==========================================================================
TEST 2   faithfulness against mask geometry, all arms on one table
         centre_distance: mask centroid's distance from frame centre
         spread: how far masked pixels sit from their own centroid
==========================================================================

  imputation: blur   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.117      +0.13321  
    random_lowres_2                  3.967  16.017      +0.04398  yes
    grad_cam_rolled                  6.150   9.262      +0.03673  
    random_lowres_4                  6.511  10.994      +0.04112  yes
    integrated_gradients             7.935   3.637      +0.07246  
    integrated_gradients_rolled     10.483   5.053      +0.00146  
    random_lowres_8                 10.897   6.229      +0.02673  yes
    random                          12.112   1.594      +0.00002  yes
    random_lowres_32                12.114   1.538      +0.01548  yes
    random_lowres_16                12.120   3.433      +0.00974  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.08923
      grad_cam_rolled              vs random_lowres_4      (spread 6.15 vs 6.51)  residual -0.00439
      integrated_gradients         vs random_lowres_4      (spread 7.93 vs 6.51)  residual +0.03134
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.48 vs 10.90)  residual -0.02527

  imputation: mean   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.117      -0.02421  
    random_lowres_2                  3.967  16.017      -0.15983  yes
    grad_cam_rolled                  6.150   9.262      -0.13944  
    random_lowres_4                  6.511  10.994      -0.12500  yes
    integrated_gradients             7.935   3.637      +0.02605  
    integrated_gradients_rolled     10.483   5.053      -0.04784  
    random_lowres_8                 10.897   6.229      -0.09769  yes
    random                          12.112   1.594      +0.00007  yes
    random_lowres_32                12.114   1.538      +0.00164  yes
    random_lowres_16                12.120   3.433      -0.07262  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.13562
      grad_cam_rolled              vs random_lowres_4      (spread 6.15 vs 6.51)  residual -0.01445
      integrated_gradients         vs random_lowres_4      (spread 7.93 vs 6.51)  residual +0.15104
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.48 vs 10.90)  residual +0.04985

==========================================================================
TEST 3   where in the removal curve the gap lives
         probability kept by the explainer's mask minus by a rolled one
==========================================================================

  blur / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.8117     0.8396   -0.0280
        0.10     0.7638     0.8190   -0.0552
        0.20     0.6833     0.7807   -0.0974
        0.30     0.6119     0.7421   -0.1302
        0.50     0.5097     0.6638   -0.1541
        0.70     0.4571     0.5768   -0.1197
        0.90     0.4351     0.4813   -0.0462
        1.00     0.4319     0.4319   +0.0000
    peak gap at fraction 0.50 (-0.1541)

  blur / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.7857     0.8430   -0.0573
        0.10     0.7534     0.8280   -0.0746
        0.20     0.7089     0.8014   -0.0926
        0.30     0.6755     0.7723   -0.0968
        0.50     0.6182     0.7097   -0.0915
        0.70     0.5607     0.6339   -0.0732
        0.90     0.4905     0.5294   -0.0389
        1.00     0.4319     0.4319   +0.0000
    peak gap at fraction 0.30 (-0.0968)

  mean / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.7038     0.7790   -0.0753
        0.10     0.5783     0.7181   -0.1398
        0.20     0.3953     0.6078   -0.2126
        0.30     0.2911     0.5062   -0.2151
        0.50     0.1912     0.3413   -0.1501
        0.70     0.1462     0.2211   -0.0749
        0.90     0.1227     0.1391   -0.0164
        1.00     0.1122     0.1122   +0.0000
    peak gap at fraction 0.30 (-0.2151)

  mean / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8562     0.8562   +0.0000
        0.05     0.5412     0.7040   -0.1628
        0.10     0.4216     0.6028   -0.1812
        0.20     0.2744     0.4399   -0.1656
        0.30     0.2070     0.3260   -0.1189
        0.50     0.1655     0.2199   -0.0544
        0.70     0.1510     0.1803   -0.0293
        0.90     0.1128     0.1196   -0.0068
        1.00     0.1122     0.1122   +0.0000
    peak gap at fraction 0.10 (-0.1812)

==========================================================================

---

## Part 4 — VALID: the three-seed replication (P3b.4)

**2026-10-18, commit `bf477ec`. Parts 3 and 4 together are what to cite.** Sessions
2 and 3 of the three planned Kaggle sessions, run as P3b.4's pre-registered
replication with `--seeds 1` and `--seeds 2`. Same config
(`configs/pilot_validity.yaml`), same model, same 1000 committed evaluation
indices, same 10 cells, 4 imputation schemes and 10 explainer arms as Part 3.
Only the seed changed.

| | seed 0 (Part 3) | seed 1 | seed 2 |
|---|---|---|---|
| commit | `34e351a` | `bf477ec` | `bf477ec` |
| units | 510, 0 failed | 510, 0 failed | 510, 0 failed |
| elapsed | 8713.3s | 8854.9s | 9065.7s |
| CIFAR-10 download in-session | 51:15 | 1:00:44 | 1:01:00 |
| epoch 1 / epoch 50 training loss | 1.5867 / 0.0016 | 1.5910 / 0.0017 | 1.5901 / 0.0016 |

Part 3 is where each quantity is defined and where the single-seed result is
worked out. Part 4 reports only what the three seeds say together. Every figure
below is read from the three logs named at the head of each subsection; the two
new logs are reproduced in full at the end of this part.

The seed axis did move. Grad-CAM's mask spread is 3.298, 3.298 and 3.295 and its
displaced copy's is 6.150, 6.188 and 6.170, so the three runs are three different
trained models; the model-free arms' spreads are identical to three decimals in
all three logs (`random_lowres_2` 3.967, `random_lowres_4` 6.511,
`random_lowres_8` 10.897), which is what an arm that reads neither model nor image
nor label should do.

### Matched faithfulness replicates: 11 of 12 cells positive and resolved in all three seeds

From `GEOMETRY-MATCHED FAITHFULNESS` in each log.

| imputation | pair | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| `black` | grad_cam vs grad_cam_rolled | +0.10068 | +0.09844 | +0.09682 |
| `black` | grad_cam vs random_lowres_4 | +0.09194 | +0.08703 | +0.08602 |
| `black` | IG vs IG_rolled | -0.00398 | -0.00554 | -0.00328 |
| `blur` | grad_cam vs grad_cam_rolled | +0.09648 | +0.09681 | +0.09676 |
| `blur` | grad_cam vs random_lowres_4 | +0.09209 | +0.09011 | +0.09094 |
| `blur` | IG vs IG_rolled | +0.07100 | +0.07172 | +0.07084 |
| `mean` | grad_cam vs grad_cam_rolled | +0.11523 | +0.11930 | +0.11554 |
| `mean` | grad_cam vs random_lowres_4 | +0.10078 | +0.10204 | +0.09337 |
| `mean` | IG vs IG_rolled | +0.07389 | +0.07790 | +0.07756 |
| `uniform_noise` | grad_cam vs grad_cam_rolled | +0.12083 | +0.12249 | +0.12748 |
| `uniform_noise` | grad_cam vs random_lowres_4 | +0.11050 | +0.11004 | +0.11395 |
| `uniform_noise` | IG vs IG_rolled | +0.04855 | +0.05236 | +0.05194 |

**What replicated:** 11 of the 12 (scheme, pair) cells are positive and resolved
in every seed, at 7.3 to 10.6 of their own bootstrap half-widths from zero, and
all 11 carry the verdict `10/10 cells beat it` in all three logs. The corrected
instrument's level statement -- an explainer beats a control of its own mask
geometry -- holds on three independent trainings.

**What did not:** `black`/IG, and it is the same cell every time. It is slightly
negative in all three seeds and unresolved in all three, at 0.62, 0.87 and 0.55
half-widths below zero. It is *consistently* unresolved rather than unstable: the
three point estimates span 0.00226, about a third of the cell's own bootstrap
half-width. The only thing that moves is the per-cell count in the verdict column --
`1/10 cells beat it` under seed 0, `NO CELL BEATS ITS MATCHED CONTROL` under
seeds 1 and 2 -- which is one cell of ten changing side on a quantity whose
interval covers zero.

### The geometry ladder replicates: 12 of 12

From `GEOMETRY LADDER` in each log. `random_lowres_<g>` is noise on a g x g lattice
upsampled as Grad-CAM is; no model, image or label enters it.

| imputation | span grid 2 -> 32, seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `black` | 0.2046 | 0.2027 | 0.2017 |
| `uniform_noise` | 0.1852 | 0.1829 | 0.1799 |
| `mean` | 0.1615 | 0.1516 | 0.1572 |
| `blur` | 0.0285 | 0.0300 | 0.0251 |

Monotone toward zero from grid 2 to grid 16 in all four schemes in all three
seeds -- **12 of 12**. Grid 32 breaks monotonicity under `blur` and only under
`blur`, in all three seeds, exactly as it did under seed 0. A model-free arm's
score is a function of how coarse its mask is, and that function is stable across
trainings.

### Raw and matched disagree in sign in the same three cells, in all three seeds

Per (scheme, explainer), the sign of the mean `faithfulness` arm from the
`FAITHFULNESS` summary against the sign of the matched reading above. Eight cells:
four schemes x two explainers.

| imputation | explainer | seed 0 raw / matched | seed 1 raw / matched | seed 2 raw / matched | |
|---|---|---|---|---|---|
| `black` | grad_cam | -0.04735 / +0.10068 | -0.05660 / +0.09844 | -0.05862 / +0.09682 | **flips** |
| `mean` | grad_cam | -0.02421 / +0.11523 | -0.01719 / +0.11930 | -0.03106 / +0.11554 | **flips** |
| `uniform_noise` | grad_cam | -0.04382 / +0.12083 | -0.04391 / +0.12249 | -0.03637 / +0.12748 | **flips** |
| `blur` | grad_cam | +0.13321 / +0.09648 | +0.13129 / +0.09681 | +0.12671 / +0.09676 | agrees |
| `black` | IG | -0.04214 / -0.00398 | -0.04681 / -0.00554 | -0.03902 / -0.00328 | agrees |
| `blur` | IG | +0.07246 / +0.07100 | +0.07165 / +0.07172 | +0.06888 / +0.07084 | agrees |
| `mean` | IG | +0.02605 / +0.07389 | +0.02631 / +0.07790 | +0.02684 / +0.07756 | agrees |
| `uniform_noise` | IG | +0.00203 / +0.04855 | +0.00221 / +0.05236 | +0.00664 / +0.05194 | agrees |

**3 of 8 cells, and the same three cells, in all three seeds.** All three are
Grad-CAM; IG never flips in any scheme in any seed. This is §2's H2a second,
structural rule -- disagreements concentrated in the explainer with the coarser
native map, Grad-CAM at 4x4 against IG at pixel resolution -- and it is the
*identity* of the flipping cells that replicates, not merely the count of three.

One qualification on the criterion. H2a counts a disagreement only when both
readings exclude zero in their own 95% CI, and `pilot_report.py` prints no
half-width on the arm-level raw mean, so the table above is a disagreement of
point-estimate signs. The per-severity half-widths it does print support two of
the three without qualification: `black`/grad_cam and `uniform_noise`/grad_cam are
resolved negative at all four severities in all three seeds. `mean`/grad_cam is
resolved negative at severities 0 and 1 in all three seeds and unresolved at
severities 3 and 5 in all three (seed 1's severity-5 reading is +0.00339 +/-
0.01715). So the flip under the primary scheme rests on its clean and
severity-1 readings, and the arm-level interval that H2a asks for has not been
computed.

### H1a: supported under two schemes, marginal under the primary scheme, and against under `blur`

This subsection is the load-bearing one, and it does not come out clean.

§2 defines **A** as the largest |mean `raw_faithfulness`| over the model-free
`random_lowres_<g>` arms and **E** as the largest |mean `matched_faithfulness`|
in the same scheme, and registers the test as a **paired bootstrap over
evaluation images of A - E**. `scripts/pilot_report.py` does not compute that
bootstrap. It reports A and E with separate intervals, and the two are correlated
-- they are built from the same images and the same removal curves -- so A - E's
half-width cannot be recovered from them. A - E is therefore bounded both ways:

- `indep` = sqrt(hw_A^2 + hw_E^2), the half-width A - E would have if the two
  readings were independent. This is the optimistic bound.
- `cons` = hw_A + hw_E, the half-width it cannot exceed whatever the correlation.
  This is the conservative bound.

A is `random_lowres_2` in all twelve (scheme, seed) combinations and its
half-width is read from the `GEOMETRY LADDER` table; E is
`grad_cam vs grad_cam_rolled` in all twelve and its half-width from
`GEOMETRY-MATCHED FAITHFULNESS`.

| scheme | seed | A | E | A - E | indep | cons | verdict |
|---|---|---|---|---|---|---|---|
| `black` | 0 | 0.1993 | 0.1007 | +0.0986 | 0.0235 | 0.0321 | supported, both bounds |
| `black` | 1 | 0.2001 | 0.0984 | +0.1016 | 0.0230 | 0.0314 | supported, both bounds |
| `black` | 2 | 0.1969 | 0.0968 | +0.1001 | 0.0242 | 0.0329 | supported, both bounds |
| `uniform_noise` | 0 | 0.1855 | 0.1208 | +0.0647 | 0.0172 | 0.0243 | supported, both bounds |
| `uniform_noise` | 1 | 0.1852 | 0.1225 | +0.0627 | 0.0177 | 0.0250 | supported, both bounds |
| `uniform_noise` | 2 | 0.1800 | 0.1275 | +0.0525 | 0.0172 | 0.0243 | supported, both bounds |
| `mean` | 0 | 0.1598 | 0.1152 | +0.0446 | 0.0220 | 0.0305 | supported, both bounds |
| `mean` | 1 | 0.1486 | 0.1193 | **+0.0293** | 0.0222 | **0.0307** | **clears `indep` only** |
| `mean` | 2 | 0.1575 | 0.1155 | +0.0419 | 0.0224 | 0.0310 | supported, both bounds |
| `blur` | 0 | 0.0440 | 0.0965 | -0.0525 | 0.0154 | 0.0218 | **against** |
| `blur` | 1 | 0.0450 | 0.0968 | -0.0518 | 0.0156 | 0.0221 | **against** |
| `blur` | 2 | 0.0390 | 0.0968 | -0.0578 | 0.0160 | 0.0227 | **against** |

**`mean` is §2's declared primary scheme** -- it is the one `configs/version_a.yaml`
runs -- so this is not a footnote. **H1a's primary-scheme test is marginal.** A - E
is positive in all three seeds, clears the optimistic bound in all three, and
fails the conservative bound in one of three. Under `black` and `uniform_noise`
it clears both bounds in every seed by a wide margin. Under `blur` it runs the
other way in every seed: E is larger than A, by more than either bound, so `blur`
is a replicated counterexample rather than noise.

**The pre-registered test would settle this and has not been run.** The paired
bootstrap of A - E needs the per-image curve data. Each session wrote its cache
to `/kaggle/working/cache` and none of the three was kept (see the operational
note below), so computing it is a re-run and not a re-analysis. Until it is run,
the correct statement is the
one above: supported under `black` and `uniform_noise`, marginal and unresolved
under `mean`, against under `blur`. §2's per-scheme-never-pooled rule is what
makes that statement sayable at all; pooling would have hidden both the `blur`
counterexample and the primary scheme's weakness.

### The matched gate fails in all three seeds

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `GATE (GEOMETRY-MATCHED)` | 0/29 | 0/28 | 0/28 |

No pair clears a half-width under 10% of its clean-to-severity-5 change on the
matched metric, in any seed. The pre-registered failure branch fires in all
three, and the P3b.4 replication adds nothing that would reopen the decision.

Part 3's n-scaling table does not itself replicate, and the direction of the
movement matters. The best ratio is 0.143, 0.143 and 0.134 across the three
seeds, so the sample size the closest pair would need is 2,045, 2,045 and 1,796
images. Scaling every resolvable pair's ratio as 1/sqrt(n):

| n | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 1,000 (actual) | 0/29 | 0/28 | 0/28 |
| 2,000 | 0/29 | 0/28 | 1/28 |
| 3,000 | 6/29 | 5/28 | 3/28 |
| 5,000 | 7/29 | 7/28 | 12/28 |
| 10,000 | 19/29 | 17/28 | 20/28 |

At n=2000 the gate passes nothing in two seeds and one pair in the third. Above
that the counts diverge enough -- 3 to 6 at n=3000, 7 to 12 at n=5000 -- that no
single n/pass-count pair should be quoted as a prediction. The claim that
survives all three seeds is the one at the sample size actually run: zero.

### The raw gate's verdict moves with the seed; the matched gate's does not

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `GATE (RAW)` | 14/85 | 16/82 | 12/87 |
| as a rate | 16.5% | 19.5% | 13.8% |
| resolvable of 120 rows | 85 | 82 | 87 |
| `GATE (GEOMETRY-MATCHED)` | 0/29 | 0/28 | 0/28 |

This is a contrast Part 3 could not draw. The raw metric is not only confounded
by mask geometry -- a model-free blob scores well below zero on it -- its *gate
verdict* is seed-unstable, moving over a 5.7-point range in pass rate on three
runs that differ in nothing but the training seed. The matched gate is stably
zero.

The membership moves too, not just the count. Across the three seeds 16 distinct
(scheme, explainer, family) rows pass at least once and only 11 pass in all
three. `black`/random_lowres_16 passes in seed 1 alone. `grad_cam` itself passes
under `black` and `mean` in seeds 0 and 1 and in neither under seed 2, while its
own displaced copy `grad_cam_rolled` passes in all three -- a gate on the raw
metric that admits the displaced map more reliably than the real one. Every pass
in every seed, all 42 of them, is in the `gaussian_noise` family.

Taken with the two gates' counts, this is the cleanest operational statement of
the measurement-validity finding: the metric whose gate sometimes passes is the
one that reads mask geometry, and the metric that reads the explainer cannot
resolve its change at this n in any seed.

### TEST 1 replicates: 4 of 4

`gain = faithfulness(rolled) - faithfulness(explainer)`; a more negative gain
means the explainer beats its own displaced copy by more. Displacement runs 5.73
to 17.90 px in five steps in all three logs.

| imputation | explainer | seed 0, 5.73 -> 17.90 px | seed 1 | seed 2 |
|---|---|---|---|---|
| `blur` | grad_cam | -0.03410 -> -0.15359, 4.5x | -0.02980 -> -0.15120, 5.1x | -0.02960 -> -0.15353, 5.2x |
| `mean` | grad_cam | -0.04357 -> -0.18904, 4.3x | -0.04502 -> -0.19522, 4.3x | -0.04722 -> -0.19531, 4.1x |
| `blur` | IG | -0.05213 -> -0.08868, 1.7x | -0.04601 -> -0.08622, 1.9x | -0.05501 -> -0.08670, 1.6x |
| `mean` | IG | -0.05582 -> -0.09381, 1.7x | -0.05516 -> -0.09716, 1.8x | -0.05617 -> -0.09853, 1.8x |

All four rows read `advantage GROWS with distance (beats its copy)` in all three
seeds -- **4 of 4 in 3 of 3**. Displacing a map further costs more, so what the
matched comparison measures is spatially specific aim and not merely a
differently shaped mask. The Grad-CAM/IG split replicates too: over the same
displacement range Grad-CAM's advantage grows by a factor of 4.1 to 5.2 and IG's
by 1.6 to 1.9, in every seed and under both imputation schemes.

### TEST 2 replicates, with its exception

TEST 2 orders every arm by mask spread and takes each real arm's residual against
the model-free arms of nearest spread.

| imputation | arm | seed 0 | seed 1 | seed 2 | comparator |
|---|---|---|---|---|---|
| `blur` | grad_cam_rolled | -0.00439 | -0.00670 | -0.00583 | `random_lowres_4`, within 0.36 of spread |
| `mean` | grad_cam_rolled | -0.01445 | -0.01726 | -0.02217 | `random_lowres_4`, within 0.36 |
| `blur` | IG_rolled | -0.02527 | -0.02653 | -0.02253 | `random_lowres_8`, within 0.43 |
| `mean` | IG_rolled | **+0.04985** | **+0.03722** | **+0.04979** | `random_lowres_8`, within 0.43 |
| `blur` | grad_cam | +0.08923 | +0.08626 | +0.08775 | `random_lowres_2`, within 0.68 |
| `mean` | grad_cam | +0.13562 | +0.13137 | +0.12642 | `random_lowres_2`, within 0.68 |
| `blur` | IG | +0.03134 | +0.03046 | +0.03310 | `random_lowres_4`, within 1.44 |
| `mean` | IG | +0.15104 | +0.14555 | +0.15127 | `random_lowres_4`, within 1.44 |

**What replicated.** `grad_cam_rolled` -- a real Grad-CAM map, displaced -- sits on
the model-free line in all three seeds, within 0.007 of it under `blur` and
within 0.023 under `mean`, while the correctly-aimed arms sit 0.03 to 0.15 above
it. A map whose geometry is intact and whose aim has been destroyed behaves like
model-free noise of its own geometry. That is the geometry account's sharpest
single prediction and it holds on three trainings.

**What did not.** `integrated_gradients_rolled` does not land on the line under
`mean`: +0.04985, +0.03722 and +0.04979, positive in all three seeds and a
quarter to a third of correctly-aimed IG's own residual in the same scheme. Under
`blur` it reads slightly below the line, -0.02527 to -0.02253, also in all three.
So the exception
replicates as reliably as the confirmation does, and the honest statement is the
one Part 3 made: clean for `grad_cam_rolled`, partial for
`integrated_gradients_rolled`.

### Calibration is consistent across seeds

Clean to severity 5, from the `CALIBRATION` block in each log.

| metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| accuracy | 0.8610 -> 0.5487 | 0.8580 -> 0.5530 | 0.8620 -> 0.5600 |
| ece_em | 0.0516 -> 0.2514 | 0.0474 -> 0.2448 | 0.0566 -> 0.2391 |
| aurc | 0.0315 -> 0.2702 | 0.0299 -> 0.2854 | 0.0264 -> 0.2707 |

The shift is doing the same thing to the model in all three seeds: a 30- to
31-point accuracy drop, a four- to fivefold rise in expected calibration error,
and an eight- to tenfold rise in AURC. Clean accuracy varies by 0.4 points across
the three trainings and severity-5 accuracy by 1.1. The degradation these runs
measure faithfulness against is not a seed artefact.

### Operational note

Neither session had the cache attached. Both logs print
`cache read  (none: /kaggle/input/shiftprofile-cache is not attached)`, as Part
3's does, so each session retrained the checkpoint from scratch and paid the
CIFAR-10 download again: `170M/170M [1:00:44<00:00, 46.8kB/s]` for seed 1 and
`[1:01:00<00:00, 46.6kB/s]` for seed 2. Roughly an hour of each session -- 2.46
and 2.52 hours of wall clock -- was a download an attached dataset would have
skipped, so the 8854.9s and 9065.7s totals are not compute figures.

The three caches are therefore independent rather than cumulative: nothing was
shared between the three sessions but the code and the committed evaluation
indices. That costs GPU time and does no harm to the results -- if anything it
makes the three runs more independent than the plan required. It has one real
cost, recorded in the H1a subsection above: no per-image curve cache survives,
so the registered paired bootstrap of A - E needs a re-run and not a
re-analysis.

### Scope

Three seeds, one model (`resnet18`), one dataset, 1000 images, one set of 10
cells. The model axis is still single; `vit_tiny` has never run. §6's
mixed-effects aggregation, which is how seeds are supposed to be combined, is not
implemented for faithfulness, so everything above is a **count across seeds with
the per-seed figures shown** and not a pooled estimate with an interval over
seeds. Read "11 of 12 in 3 of 3" as what it says: eleven cells held in three
independent runs. It is not a variance component.

Raw output for both sessions below, sliced from line 89 of each log to its end by
script -- not transcribed -- from
`docs/runs/2026-10-18_seed1.log` and `docs/runs/2026-10-18_seed2.log`.

### Seed 1 raw log

  curves: 400 units, 2964s (7.4s/unit), at curves:resnet18:1:fog:5:random_lowres_32:black

============================================================
Completed 510 cells, skipped 0 cached, failed 0 cells
Elapsed time: 8854.9s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean, blur, uniform_noise, black
explainers  integrated_gradients, grad_cam, random, integrated_gradients_rolled, grad_cam_rolled, random_lowres_2, random_lowres_4, random_lowres_8, random_lowres_16, random_lowres_32

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8580    0.2090    0.0474    0.0559    0.0299
   1     0.8297    0.2483    0.0663    0.0841    0.0410
   3     0.7267    0.4099    0.1290    0.1413    0.1157
   5     0.5530    0.6813    0.2448    0.2615    0.2854

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================

  imputation: black
    grad_cam               mean -0.05660   2/10 cells above 0  
    grad_cam_rolled        mean -0.15504   1/10 cells above 0  
    integrated_gradients   mean -0.04681   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.04127   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00004   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.03833   2/10 cells above 0  
    random_lowres_2        mean -0.20005   1/10 cells above 0  
    random_lowres_32       mean +0.00261   2/10 cells above 0  
    random_lowres_4        mean -0.14363   1/10 cells above 0  
    random_lowres_8        mean -0.07938   2/10 cells above 0  

  imputation: blur
    grad_cam               mean +0.13129   10/10 cells above 0  
    grad_cam_rolled        mean +0.03448   8/10 cells above 0  
    integrated_gradients   mean +0.07165   10/10 cells above 0  
    integrated_gradients_rolled mean -0.00008   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean +0.00698   3/10 cells above 0  
    random_lowres_2        mean +0.04503   9/10 cells above 0  
    random_lowres_32       mean +0.01504   8/10 cells above 0  
    random_lowres_4        mean +0.04118   9/10 cells above 0  
    random_lowres_8        mean +0.02645   8/10 cells above 0  

  imputation: mean
    grad_cam               mean -0.01719   2/10 cells above 0  
    grad_cam_rolled        mean -0.13650   1/10 cells above 0  
    integrated_gradients   mean +0.02631   8/10 cells above 0  
    integrated_gradients_rolled mean -0.05159   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00001   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.07261   1/10 cells above 0  
    random_lowres_2        mean -0.14856   1/10 cells above 0  
    random_lowres_32       mean +0.00307   2/10 cells above 0  
    random_lowres_4        mean -0.11924   2/10 cells above 0  
    random_lowres_8        mean -0.08881   2/10 cells above 0  

  imputation: uniform_noise
    grad_cam               mean -0.04391   1/10 cells above 0  
    grad_cam_rolled        mean -0.16640   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients   mean +0.00221   3/10 cells above 0  
    integrated_gradients_rolled mean -0.05015   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.07386   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_2        mean -0.18519   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean -0.00226   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.15395   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_8        mean -0.11965   0/10 cells above 0  NO CELL BEATS RANDOM

imputation      explainer               sev   faithfulness   half_width
black           grad_cam                  0       -0.08366      0.01871
black           grad_cam                  1       -0.07474      0.01919
black           grad_cam                  3       -0.03620      0.02045
black           grad_cam                  5       -0.04984      0.02027
black           grad_cam_rolled           0       -0.20956      0.01905
black           grad_cam_rolled           1       -0.19749      0.01897
black           grad_cam_rolled           3       -0.13520      0.02063
black           grad_cam_rolled           5       -0.11425      0.02034
black           integrated_gradients      0       -0.06603      0.00779
black           integrated_gradients      1       -0.05921      0.00742
black           integrated_gradients      3       -0.04384      0.00713
black           integrated_gradients      5       -0.03098      0.00692
black           integrated_gradients_rolled    0       -0.05822      0.00739
black           integrated_gradients_rolled    1       -0.05360      0.00704
black           integrated_gradients_rolled    3       -0.03651      0.00711
black           integrated_gradients_rolled    5       -0.02804      0.00687
black           random                    0       +0.00007      0.00013
black           random                    1       +0.00004      0.00013
black           random                    3       +0.00001      0.00012
black           random                    5       +0.00007      0.00014
black           random_lowres_16          0       -0.07407      0.01283
black           random_lowres_16          1       -0.06345      0.01248
black           random_lowres_16          3       -0.02845      0.01299
black           random_lowres_16          5       -0.01118      0.01140
black           random_lowres_2           0       -0.24354      0.01880
black           random_lowres_2           1       -0.23844      0.01880
black           random_lowres_2           3       -0.18373      0.02092
black           random_lowres_2           5       -0.16349      0.02044
black           random_lowres_32          0       -0.00122      0.00557
black           random_lowres_32          1       -0.00008      0.00532
black           random_lowres_32          3       +0.00444      0.00525
black           random_lowres_32          5       +0.00476      0.00487
black           random_lowres_4           0       -0.19149      0.01863
black           random_lowres_4           1       -0.18064      0.01871
black           random_lowres_4           3       -0.12258      0.02030
black           random_lowres_4           5       -0.11173      0.01970
black           random_lowres_8           0       -0.12970      0.01606
black           random_lowres_8           1       -0.11666      0.01621
black           random_lowres_8           3       -0.06334      0.01748
black           random_lowres_8           5       -0.04137      0.01590
blur            grad_cam                  0       +0.12781      0.01309
blur            grad_cam                  1       +0.13276      0.01302
blur            grad_cam                  3       +0.14102      0.01312
blur            grad_cam                  5       +0.12126      0.01225
blur            grad_cam_rolled           0       +0.03185      0.01108
blur            grad_cam_rolled           1       +0.03132      0.01120
blur            grad_cam_rolled           3       +0.03929      0.01189
blur            grad_cam_rolled           5       +0.03371      0.01150
blur            integrated_gradients      0       +0.06356      0.00847
blur            integrated_gradients      1       +0.06674      0.00905
blur            integrated_gradients      3       +0.07989      0.00993
blur            integrated_gradients      5       +0.07100      0.00959
blur            integrated_gradients_rolled    0       -0.00102      0.00766
blur            integrated_gradients_rolled    1       -0.00221      0.00789
blur            integrated_gradients_rolled    3       +0.00193      0.00846
blur            integrated_gradients_rolled    5       +0.00036      0.00816
blur            random                    0       +0.00001      0.00008
blur            random                    1       -0.00000      0.00009
blur            random                    3       +0.00001      0.00011
blur            random                    5       -0.00006      0.00013
blur            random_lowres_16          0       +0.00645      0.00822
blur            random_lowres_16          1       +0.00653      0.00878
blur            random_lowres_16          3       +0.00593      0.00909
blur            random_lowres_16          5       +0.00864      0.00918
blur            random_lowres_2           0       +0.04690      0.01087
blur            random_lowres_2           1       +0.04555      0.01095
blur            random_lowres_2           3       +0.04698      0.01140
blur            random_lowres_2           5       +0.04195      0.01091
blur            random_lowres_32          0       +0.01957      0.00636
blur            random_lowres_32          1       +0.01977      0.00685
blur            random_lowres_32          3       +0.01464      0.00687
blur            random_lowres_32          5       +0.00920      0.00697
blur            random_lowres_4           0       +0.04043      0.01057
blur            random_lowres_4           1       +0.04013      0.01092
blur            random_lowres_4           3       +0.04260      0.01141
blur            random_lowres_4           5       +0.04107      0.01092
blur            random_lowres_8           0       +0.02276      0.01006
blur            random_lowres_8           1       +0.02439      0.01020
blur            random_lowres_8           3       +0.02841      0.01119
blur            random_lowres_8           5       +0.02778      0.01062
mean            grad_cam                  0       -0.05091      0.01703
mean            grad_cam                  1       -0.04058      0.01749
mean            grad_cam                  3       -0.00316      0.01861
mean            grad_cam                  5       +0.00339      0.01715
mean            grad_cam_rolled           0       -0.18872      0.01891
mean            grad_cam_rolled           1       -0.17664      0.01940
mean            grad_cam_rolled           3       -0.12364      0.01995
mean            grad_cam_rolled           5       -0.09181      0.01971
mean            integrated_gradients      0       +0.02519      0.00707
mean            integrated_gradients      1       +0.02920      0.00732
mean            integrated_gradients      3       +0.02712      0.00747
mean            integrated_gradients      5       +0.02298      0.00732
mean            integrated_gradients_rolled    0       -0.06717      0.00818
mean            integrated_gradients_rolled    1       -0.06106      0.00837
mean            integrated_gradients_rolled    3       -0.04993      0.00820
mean            integrated_gradients_rolled    5       -0.03860      0.00818
mean            random                    0       -0.00000      0.00015
mean            random                    1       +0.00002      0.00013
mean            random                    3       -0.00005      0.00015
mean            random                    5       -0.00001      0.00014
mean            random_lowres_16          0       -0.10762      0.01323
mean            random_lowres_16          1       -0.09781      0.01282
mean            random_lowres_16          3       -0.06653      0.01311
mean            random_lowres_16          5       -0.04181      0.01236
mean            random_lowres_2           0       -0.19685      0.01785
mean            random_lowres_2           1       -0.18591      0.01846
mean            random_lowres_2           3       -0.13635      0.01948
mean            random_lowres_2           5       -0.10734      0.01858
mean            random_lowres_32          0       -0.00190      0.00620
mean            random_lowres_32          1       +0.00092      0.00603
mean            random_lowres_32          3       +0.00528      0.00624
mean            random_lowres_32          5       +0.00466      0.00599
mean            random_lowres_4           0       -0.17079      0.01847
mean            random_lowres_4           1       -0.15685      0.01856
mean            random_lowres_4           3       -0.10547      0.01948
mean            random_lowres_4           5       -0.07822      0.01883
mean            random_lowres_8           0       -0.13628      0.01723
mean            random_lowres_8           1       -0.12274      0.01733
mean            random_lowres_8           3       -0.07940      0.01785
mean            random_lowres_8           5       -0.04849      0.01661
uniform_noise   grad_cam                  0       -0.06664      0.00975
uniform_noise   grad_cam                  1       -0.05952      0.00945
uniform_noise   grad_cam                  3       -0.03921      0.00945
uniform_noise   grad_cam                  5       -0.02543      0.00871
uniform_noise   grad_cam_rolled           0       -0.21487      0.01342
uniform_noise   grad_cam_rolled           1       -0.20161      0.01369
uniform_noise   grad_cam_rolled           3       -0.15805      0.01361
uniform_noise   grad_cam_rolled           5       -0.12338      0.01347
uniform_noise   integrated_gradients      0       +0.00329      0.00571
uniform_noise   integrated_gradients      1       +0.00526      0.00586
uniform_noise   integrated_gradients      3       +0.00016      0.00591
uniform_noise   integrated_gradients      5       +0.00087      0.00570
uniform_noise   integrated_gradients_rolled    0       -0.06163      0.00627
uniform_noise   integrated_gradients_rolled    1       -0.05863      0.00637
uniform_noise   integrated_gradients_rolled    3       -0.04755      0.00634
uniform_noise   integrated_gradients_rolled    5       -0.04043      0.00632
uniform_noise   random                    0       -0.00001      0.00011
uniform_noise   random                    1       -0.00004      0.00012
uniform_noise   random                    3       -0.00001      0.00012
uniform_noise   random                    5       -0.00001      0.00012
uniform_noise   random_lowres_16          0       -0.09556      0.00853
uniform_noise   random_lowres_16          1       -0.09072      0.00822
uniform_noise   random_lowres_16          3       -0.07047      0.00821
uniform_noise   random_lowres_16          5       -0.05315      0.00781
uniform_noise   random_lowres_2           0       -0.22840      0.01285
uniform_noise   random_lowres_2           1       -0.22206      0.01285
uniform_noise   random_lowres_2           3       -0.17656      0.01327
uniform_noise   random_lowres_2           5       -0.14255      0.01258
uniform_noise   random_lowres_32          0       -0.00124      0.00426
uniform_noise   random_lowres_32          1       -0.00269      0.00429
uniform_noise   random_lowres_32          3       -0.00190      0.00430
uniform_noise   random_lowres_32          5       -0.00253      0.00417
uniform_noise   random_lowres_4           0       -0.19809      0.01290
uniform_noise   random_lowres_4           1       -0.18892      0.01264
uniform_noise   random_lowres_4           3       -0.14550      0.01330
uniform_noise   random_lowres_4           5       -0.11273      0.01260
uniform_noise   random_lowres_8           0       -0.15866      0.01098
uniform_noise   random_lowres_8           1       -0.14962      0.01114
uniform_noise   random_lowres_8           3       -0.11239      0.01112
uniform_noise   random_lowres_8           5       -0.08394      0.01056

========================================================================
GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid
                  lattice, upsampled as Grad-CAM is. No model input.
========================================================================
imputation       grid   faithfulness   half_width
black               2       -0.20005      0.01993
black               4       -0.14363      0.01948
black               8       -0.07938      0.01648
black              16       -0.03833      0.01234
black              32       +0.00261      0.00519
blur                2       +0.04503      0.01106
blur                4       +0.04118      0.01103
blur                8       +0.02645      0.01061
blur               16       +0.00698      0.00894
blur               32       +0.01504      0.00684
mean                2       -0.14856      0.01874
mean                4       -0.11924      0.01891
mean                8       -0.08881      0.01726
mean               16       -0.07261      0.01281
mean               32       +0.00307      0.00610
uniform_noise       2       -0.18519      0.01289
uniform_noise       4       -0.15395      0.01285
uniform_noise       8       -0.11965      0.01095
uniform_noise      16       -0.07386      0.00812
uniform_noise      32       -0.00226      0.00425

Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on this
curve at grid=4, its score is its resolution and not its content.

========================================================================
GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the
                                same mask geometry; >0 beats it
========================================================================
imputation      explainer                vs                   kind          matched     half_w  verdict
black           grad_cam                 grad_cam_rolled      alignment    +0.09844    0.01145  10/10 cells beat it
black           grad_cam                 random_lowres_4      geometry     +0.08703    0.01078  10/10 cells beat it
black           integrated_gradients     integrated_gradients_rolled alignment    -0.00554    0.00634  NO CELL BEATS ITS MATCHED CONTROL
blur            grad_cam                 grad_cam_rolled      alignment    +0.09681    0.01102  10/10 cells beat it
blur            grad_cam                 random_lowres_4      geometry     +0.09011    0.01014  10/10 cells beat it
blur            integrated_gradients     integrated_gradients_rolled alignment    +0.07172    0.00971  10/10 cells beat it
mean            grad_cam                 grad_cam_rolled      alignment    +0.11930    0.01196  10/10 cells beat it
mean            grad_cam                 random_lowres_4      geometry     +0.10204    0.01113  10/10 cells beat it
mean            integrated_gradients     integrated_gradients_rolled alignment    +0.07790    0.00811  10/10 cells beat it
uniform_noise   grad_cam                 grad_cam_rolled      alignment    +0.12249    0.01209  10/10 cells beat it
uniform_noise   grad_cam                 random_lowres_4      geometry     +0.11004    0.01137  10/10 cells beat it
uniform_noise   integrated_gradients     integrated_gradients_rolled alignment    +0.05236    0.00687  10/10 cells beat it

The pixel-i.i.d. control cancels exactly in this difference: both terms
use the same control_seed over the same images, so what is left is the
explainer scored against its own geometry.

========================================================================
P3 GATE (RAW)   half-width < 10% of the clean-to-severity-5 change
                on `faithfulness`: the explainer against a pixel-iid
                control. The validity run showed that quantity tracks
                mask geometry -- a model-free blob scores well below
                zero on it -- so this table is reported, not read.
========================================================================
imputation      explainer                   family               change     half_w     ratio  verdict
black           grad_cam                    defocus_blur       -0.10551    0.01890     0.179  FAIL
black           grad_cam                    fog                -0.04460    0.01936     0.434  FAIL
black           grad_cam                    gaussian_noise     +0.25157    0.02256     0.090  PASS
black           grad_cam_rolled             defocus_blur       -0.02781    0.01905     0.685  FAIL
black           grad_cam_rolled             fog                +0.03932    0.01905     0.485  FAIL
black           grad_cam_rolled             gaussian_noise     +0.27442    0.02432     0.089  PASS
black           integrated_gradients        defocus_blur       +0.01047    0.00779     0.744  FAIL
black           integrated_gradients        fog                +0.03620    0.00779     0.215  FAIL
black           integrated_gradients        gaussian_noise     +0.05847    0.00779     0.133  FAIL
black           integrated_gradients_rolled defocus_blur       +0.00519    0.00739       n/a  no change
black           integrated_gradients_rolled fog                +0.02327    0.00739     0.317  FAIL
black           integrated_gradients_rolled gaussian_noise     +0.06207    0.00786     0.127  FAIL
black           random                      defocus_blur       +0.00005    0.00014       n/a  no change
black           random                      fog                -0.00001    0.00013       n/a  no change
black           random                      gaussian_noise     -0.00005    0.00019       n/a  no change
black           random_lowres_16            defocus_blur       +0.00324    0.01283       n/a  no change
black           random_lowres_16            fog                +0.04705    0.01283     0.273  FAIL
black           random_lowres_16            gaussian_noise     +0.13840    0.01375     0.099  PASS
black           random_lowres_2             defocus_blur       -0.04869    0.01880     0.386  FAIL
black           random_lowres_2             fog                +0.00783    0.01964       n/a  no change
black           random_lowres_2             gaussian_noise     +0.28103    0.02362     0.084  PASS
black           random_lowres_32            defocus_blur       -0.00895    0.00557     0.622  FAIL
black           random_lowres_32            fog                -0.00122    0.00557       n/a  no change
black           random_lowres_32            gaussian_noise     +0.02812    0.00669     0.238  FAIL
black           random_lowres_4             defocus_blur       -0.03931    0.01863     0.474  FAIL
black           random_lowres_4             fog                +0.01830    0.01897       n/a  no change
black           random_lowres_4             gaussian_noise     +0.26028    0.02299     0.088  PASS
black           random_lowres_8             defocus_blur       -0.00285    0.01606       n/a  no change
black           random_lowres_8             fog                +0.05276    0.01606     0.304  FAIL
black           random_lowres_8             gaussian_noise     +0.21509    0.01967     0.091  PASS
blur            grad_cam                    defocus_blur       -0.06914    0.01309     0.189  FAIL
blur            grad_cam                    fog                -0.01819    0.01309     0.719  FAIL
blur            grad_cam                    gaussian_noise     +0.06770    0.01429     0.211  FAIL
blur            grad_cam_rolled             defocus_blur       -0.03070    0.01108     0.361  FAIL
blur            grad_cam_rolled             fog                -0.02714    0.01108     0.408  FAIL
blur            grad_cam_rolled             gaussian_noise     +0.06342    0.01388     0.219  FAIL
blur            integrated_gradients        defocus_blur       -0.00479    0.00847       n/a  no change
blur            integrated_gradients        fog                +0.03409    0.00884     0.259  FAIL
blur            integrated_gradients        gaussian_noise     -0.00699    0.01149       n/a  no change
blur            integrated_gradients_rolled defocus_blur       -0.00442    0.00766       n/a  no change
blur            integrated_gradients_rolled fog                -0.00134    0.00766       n/a  no change
blur            integrated_gradients_rolled gaussian_noise     +0.00991    0.01031       n/a  no change
blur            random                      defocus_blur       -0.00001    0.00008       n/a  no change
blur            random                      fog                -0.00006    0.00014       n/a  no change
blur            random                      gaussian_noise     -0.00015    0.00017       n/a  no change
blur            random_lowres_16            defocus_blur       -0.01480    0.00822     0.555  FAIL
blur            random_lowres_16            fog                -0.00721    0.00822       n/a  no change
blur            random_lowres_16            gaussian_noise     +0.02859    0.01232     0.431  FAIL
blur            random_lowres_2             defocus_blur       -0.03804    0.01087     0.286  FAIL
blur            random_lowres_2             fog                -0.02954    0.01087     0.368  FAIL
blur            random_lowres_2             gaussian_noise     +0.05271    0.01319     0.250  FAIL
blur            random_lowres_32            defocus_blur       -0.01512    0.00636     0.421  FAIL
blur            random_lowres_32            fog                -0.01752    0.00636     0.363  FAIL
blur            random_lowres_32            gaussian_noise     +0.00156    0.00968       n/a  no change
blur            random_lowres_4             defocus_blur       -0.03506    0.01057     0.301  FAIL
blur            random_lowres_4             fog                -0.02692    0.01057     0.393  FAIL
blur            random_lowres_4             gaussian_noise     +0.06390    0.01379     0.216  FAIL
blur            random_lowres_8             defocus_blur       -0.02152    0.01006     0.467  FAIL
blur            random_lowres_8             fog                -0.01384    0.01006     0.727  FAIL
blur            random_lowres_8             gaussian_noise     +0.05040    0.01376     0.273  FAIL
mean            grad_cam                    defocus_blur       -0.04692    0.01703     0.363  FAIL
mean            grad_cam                    fog                -0.01931    0.01703     0.882  FAIL
mean            grad_cam                    gaussian_noise     +0.22914    0.02120     0.093  PASS
mean            grad_cam_rolled             defocus_blur       -0.01353    0.01891       n/a  no change
mean            grad_cam_rolled             fog                +0.02164    0.01891     0.874  FAIL
mean            grad_cam_rolled             gaussian_noise     +0.28262    0.02529     0.089  PASS
mean            integrated_gradients        defocus_blur       -0.02652    0.00707     0.267  FAIL
mean            integrated_gradients        fog                -0.01194    0.00707     0.593  FAIL
mean            integrated_gradients        gaussian_noise     +0.03183    0.00842     0.264  FAIL
mean            integrated_gradients_rolled defocus_blur       +0.00880    0.00818     0.929  FAIL
mean            integrated_gradients_rolled fog                +0.01216    0.00840     0.691  FAIL
mean            integrated_gradients_rolled gaussian_noise     +0.06473    0.00827     0.128  FAIL
mean            random                      defocus_blur       -0.00001    0.00015       n/a  no change
mean            random                      fog                +0.00002    0.00015       n/a  no change
mean            random                      gaussian_noise     -0.00003    0.00015       n/a  no change
mean            random_lowres_16            defocus_blur       +0.00852    0.01323       n/a  no change
mean            random_lowres_16            fog                +0.03361    0.01323     0.394  FAIL
mean            random_lowres_16            gaussian_noise     +0.15532    0.01534     0.099  PASS
mean            random_lowres_2             defocus_blur       -0.01499    0.01785       n/a  no change
mean            random_lowres_2             fog                +0.00953    0.01785       n/a  no change
mean            random_lowres_2             gaussian_noise     +0.27399    0.02342     0.085  PASS
mean            random_lowres_32            defocus_blur       -0.00281    0.00620       n/a  no change
mean            random_lowres_32            fog                +0.00544    0.00620       n/a  no change
mean            random_lowres_32            gaussian_noise     +0.01704    0.00631     0.370  FAIL
mean            random_lowres_4             defocus_blur       -0.01719    0.01847       n/a  no change
mean            random_lowres_4             fog                +0.01328    0.01847       n/a  no change
mean            random_lowres_4             gaussian_noise     +0.28162    0.02401     0.085  PASS
mean            random_lowres_8             defocus_blur       +0.00075    0.01723       n/a  no change
mean            random_lowres_8             fog                +0.02351    0.01723     0.733  FAIL
mean            random_lowres_8             gaussian_noise     +0.23911    0.02077     0.087  PASS
uniform_noise   grad_cam                    defocus_blur       +0.02365    0.00975     0.412  FAIL
uniform_noise   grad_cam                    fog                +0.02103    0.00975     0.464  FAIL
uniform_noise   grad_cam                    gaussian_noise     +0.07896    0.00975     0.123  FAIL
uniform_noise   grad_cam_rolled             defocus_blur       +0.05142    0.01342     0.261  FAIL
uniform_noise   grad_cam_rolled             fog                +0.06065    0.01342     0.221  FAIL
uniform_noise   grad_cam_rolled             gaussian_noise     +0.16240    0.01505     0.093  PASS
uniform_noise   integrated_gradients        defocus_blur       -0.01760    0.00573     0.326  FAIL
uniform_noise   integrated_gradients        fog                -0.01500    0.00571     0.381  FAIL
uniform_noise   integrated_gradients        gaussian_noise     +0.02534    0.00571     0.225  FAIL
uniform_noise   integrated_gradients_rolled defocus_blur       +0.00922    0.00639     0.693  FAIL
uniform_noise   integrated_gradients_rolled fog                +0.00609    0.00640       n/a  no change
uniform_noise   integrated_gradients_rolled gaussian_noise     +0.04828    0.00627     0.130  FAIL
uniform_noise   random                      defocus_blur       -0.00007    0.00014       n/a  no change
uniform_noise   random                      fog                +0.00007    0.00012       n/a  no change
uniform_noise   random                      gaussian_noise     -0.00000    0.00011       n/a  no change
uniform_noise   random_lowres_16            defocus_blur       +0.02023    0.00853     0.422  FAIL
uniform_noise   random_lowres_16            fog                +0.03591    0.00853     0.238  FAIL
uniform_noise   random_lowres_16            gaussian_noise     +0.07109    0.00859     0.121  FAIL
uniform_noise   random_lowres_2             defocus_blur       +0.04619    0.01285     0.278  FAIL
uniform_noise   random_lowres_2             fog                +0.05062    0.01285     0.254  FAIL
uniform_noise   random_lowres_2             gaussian_noise     +0.16074    0.01412     0.088  PASS
uniform_noise   random_lowres_32            defocus_blur       -0.00059    0.00426       n/a  no change
uniform_noise   random_lowres_32            fog                -0.00290    0.00426       n/a  no change
uniform_noise   random_lowres_32            gaussian_noise     -0.00038    0.00464       n/a  no change
uniform_noise   random_lowres_4             defocus_blur       +0.04475    0.01290     0.288  FAIL
uniform_noise   random_lowres_4             fog                +0.05831    0.01290     0.221  FAIL
uniform_noise   random_lowres_4             gaussian_noise     +0.15303    0.01416     0.093  PASS
uniform_noise   random_lowres_8             defocus_blur       +0.04143    0.01098     0.265  FAIL
uniform_noise   random_lowres_8             fog                +0.05815    0.01098     0.189  FAIL
uniform_noise   random_lowres_8             gaussian_noise     +0.12457    0.01165     0.094  PASS

GATE (RAW): 16/82 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================

========================================================================
P3 GATE (GEOMETRY-MATCHED)
   half-width < 10% of the clean-to-severity-5 change in `matched`:
   the explainer minus a control of its OWN mask geometry.
   THIS is the gate the 1000-vs-2000 decision reads. The raw
   gate above gates a quantity the validity run showed
   confounded, so neither its passes nor its failures bear on
   that decision.
========================================================================
imputation      explainer vs reference                              family               change     half_w     ratio  verdict
black           grad_cam vs grad_cam_rolled                         defocus_blur       -0.07770    0.01223     0.157  FAIL
black           grad_cam vs grad_cam_rolled                         fog                -0.08392    0.01223     0.146  FAIL
black           grad_cam vs grad_cam_rolled                         gaussian_noise     -0.02285    0.01223     0.535  FAIL
black           grad_cam vs random_lowres_4                         defocus_blur       -0.06620    0.01116     0.169  FAIL
black           grad_cam vs random_lowres_4                         fog                -0.06291    0.01116     0.177  FAIL
black           grad_cam vs random_lowres_4                         gaussian_noise     -0.00871    0.01116       n/a  no change
black           integrated_gradients vs integrated_gradients_rolled defocus_blur       +0.00528    0.00714       n/a  no change
black           integrated_gradients vs integrated_gradients_rolled fog                +0.01292    0.00714     0.553  FAIL
black           integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.00360    0.00714       n/a  no change
blur            grad_cam vs grad_cam_rolled                         defocus_blur       -0.03845    0.01101     0.286  FAIL
blur            grad_cam vs grad_cam_rolled                         fog                +0.00895    0.01101       n/a  no change
blur            grad_cam vs grad_cam_rolled                         gaussian_noise     +0.00428    0.01192       n/a  no change
blur            grad_cam vs random_lowres_4                         defocus_blur       -0.03408    0.01008     0.296  FAIL
blur            grad_cam vs random_lowres_4                         fog                +0.00873    0.01008       n/a  no change
blur            grad_cam vs random_lowres_4                         gaussian_noise     +0.00380    0.01172       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.00037    0.00908       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled fog                +0.03543    0.00944     0.266  FAIL
blur            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.01690    0.01146     0.678  FAIL
mean            grad_cam vs grad_cam_rolled                         defocus_blur       -0.03338    0.01224     0.367  FAIL
mean            grad_cam vs grad_cam_rolled                         fog                -0.04095    0.01224     0.299  FAIL
mean            grad_cam vs grad_cam_rolled                         gaussian_noise     -0.05348    0.01224     0.229  FAIL
mean            grad_cam vs random_lowres_4                         defocus_blur       -0.02973    0.01152     0.388  FAIL
mean            grad_cam vs random_lowres_4                         fog                -0.03260    0.01152     0.353  FAIL
mean            grad_cam vs random_lowres_4                         gaussian_noise     -0.05248    0.01152     0.220  FAIL
mean            integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.03532    0.00814     0.230  FAIL
mean            integrated_gradients vs integrated_gradients_rolled fog                -0.02410    0.00814     0.338  FAIL
mean            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.03290    0.00819     0.249  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         defocus_blur       -0.02777    0.01197     0.431  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         fog                -0.03962    0.01270     0.320  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         gaussian_noise     -0.08344    0.01197     0.143  FAIL
uniform_noise   grad_cam vs random_lowres_4                         defocus_blur       -0.02110    0.01139     0.540  FAIL
uniform_noise   grad_cam vs random_lowres_4                         fog                -0.03728    0.01139     0.306  FAIL
uniform_noise   grad_cam vs random_lowres_4                         gaussian_noise     -0.07407    0.01139     0.154  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.02682    0.00699     0.261  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled fog                -0.02110    0.00710     0.337  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.02294    0.00699     0.305  FAIL

GATE (GEOMETRY-MATCHED): 0/28 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
imputation  mean, blur
pairs       integrated_gradients vs integrated_gradients_rolled, grad_cam vs grad_cam_rolled
roll seed   0

==========================================================================
TEST 1   does displacing the map FURTHER cost more?
         gain = faithfulness(rolled) - faithfulness(explainer)
         gain < 0 means the explainer beats its own displaced copy;
         |gain| growing with distance means its aim is spatially
         specific, not merely a differently shaped mask
==========================================================================
imputation      explainer              displ_px       gain    half_w      n
mean            integrated_gradients       5.73   -0.05516   0.00511   2040
mean            integrated_gradients       9.82   -0.06393   0.00543   2050
mean            integrated_gradients      12.56   -0.07908   0.00562   2100
mean            integrated_gradients      14.78   -0.09220   0.00608   2100
mean            integrated_gradients      17.90   -0.09716   0.00590   2030
mean            grad_cam                   5.73   -0.04502   0.00578   2040
mean            grad_cam                   9.82   -0.08319   0.00790   2050
mean            grad_cam                  12.56   -0.11710   0.00805   2100
mean            grad_cam                  14.78   -0.15320   0.00877   2100
mean            grad_cam                  17.90   -0.19522   0.00980   2030
blur            integrated_gradients       5.73   -0.04601   0.00619   2040
blur            integrated_gradients       9.82   -0.06652   0.00663   2050
blur            integrated_gradients      12.56   -0.07036   0.00697   2100
blur            integrated_gradients      14.78   -0.09083   0.00696   2100
blur            integrated_gradients      17.90   -0.08622   0.00734   2030
blur            grad_cam                   5.73   -0.02980   0.00491   2040
blur            grad_cam                   9.82   -0.07327   0.00704   2050
blur            grad_cam                  12.56   -0.10272   0.00789   2100
blur            grad_cam                  14.78   -0.12464   0.00819   2100
blur            grad_cam                  17.90   -0.15120   0.00988   2030

  blur            grad_cam               -0.02980 -> -0.15120  |swing| +0.12140 vs half-width 0.00988   advantage GROWS with distance (beats its copy)
  blur            integrated_gradients   -0.04601 -> -0.08622  |swing| +0.04021 vs half-width 0.00734   advantage GROWS with distance (beats its copy)
  mean            grad_cam               -0.04502 -> -0.19522  |swing| +0.15020 vs half-width 0.00980   advantage GROWS with distance (beats its copy)
  mean            integrated_gradients   -0.05516 -> -0.09716  |swing| +0.04200 vs half-width 0.00590   advantage GROWS with distance (beats its copy)

==========================================================================
TEST 2   faithfulness against mask geometry, all arms on one table
         centre_distance: mask centroid's distance from frame centre
         spread: how far masked pixels sit from their own centroid
==========================================================================

  imputation: blur   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.128      +0.13129  
    random_lowres_2                  3.967  16.017      +0.04503  yes
    grad_cam_rolled                  6.188   9.265      +0.03448  
    random_lowres_4                  6.511  10.994      +0.04118  yes
    integrated_gradients             7.867   3.667      +0.07165  
    integrated_gradients_rolled     10.470   5.117      -0.00008  
    random_lowres_8                 10.897   6.229      +0.02645  yes
    random                          12.112   1.594      -0.00002  yes
    random_lowres_32                12.114   1.538      +0.01504  yes
    random_lowres_16                12.120   3.433      +0.00698  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.08626
      grad_cam_rolled              vs random_lowres_4      (spread 6.19 vs 6.51)  residual -0.00670
      integrated_gradients         vs random_lowres_4      (spread 7.87 vs 6.51)  residual +0.03046
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.47 vs 10.90)  residual -0.02653

  imputation: mean   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.298   6.128      -0.01719  
    random_lowres_2                  3.967  16.017      -0.14856  yes
    grad_cam_rolled                  6.188   9.265      -0.13650  
    random_lowres_4                  6.511  10.994      -0.11924  yes
    integrated_gradients             7.867   3.667      +0.02631  
    integrated_gradients_rolled     10.470   5.117      -0.05159  
    random_lowres_8                 10.897   6.229      -0.08881  yes
    random                          12.112   1.594      -0.00001  yes
    random_lowres_32                12.114   1.538      +0.00307  yes
    random_lowres_16                12.120   3.433      -0.07261  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.30 vs 3.97)  residual +0.13137
      grad_cam_rolled              vs random_lowres_4      (spread 6.19 vs 6.51)  residual -0.01726
      integrated_gradients         vs random_lowres_4      (spread 7.87 vs 6.51)  residual +0.14555
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.47 vs 10.90)  residual +0.03722

==========================================================================
TEST 3   where in the removal curve the gap lives
         probability kept by the explainer's mask minus by a rolled one
==========================================================================

  blur / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8526     0.8526   +0.0000
        0.05     0.8102     0.8357   -0.0255
        0.10     0.7632     0.8184   -0.0553
        0.20     0.6787     0.7821   -0.1035
        0.30     0.6086     0.7412   -0.1326
        0.50     0.5040     0.6581   -0.1541
        0.70     0.4513     0.5656   -0.1143
        0.90     0.4274     0.4773   -0.0498
        1.00     0.4245     0.4245   +0.0000
    peak gap at fraction 0.50 (-0.1541)

  blur / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8526     0.8526   +0.0000
        0.05     0.7777     0.8393   -0.0616
        0.10     0.7456     0.8260   -0.0804
        0.20     0.7000     0.7981   -0.0981
        0.30     0.6658     0.7674   -0.1016
        0.50     0.6102     0.7012   -0.0909
        0.70     0.5586     0.6281   -0.0696
        0.90     0.4886     0.5250   -0.0364
        1.00     0.4245     0.4245   +0.0000
    peak gap at fraction 0.30 (-0.1016)

  mean / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8526     0.8526   +0.0000
        0.05     0.7009     0.7758   -0.0749
        0.10     0.5729     0.7225   -0.1495
        0.20     0.3866     0.6196   -0.2329
        0.30     0.2812     0.5157   -0.2345
        0.50     0.1934     0.3374   -0.1440
        0.70     0.1421     0.2119   -0.0698
        0.90     0.1141     0.1348   -0.0207
        1.00     0.1030     0.1030   +0.0000
    peak gap at fraction 0.30 (-0.2345)

  mean / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8526     0.8526   +0.0000
        0.05     0.5417     0.7118   -0.1701
        0.10     0.4196     0.6161   -0.1965
        0.20     0.2783     0.4526   -0.1744
        0.30     0.2129     0.3352   -0.1223
        0.50     0.1716     0.2269   -0.0552
        0.70     0.1539     0.1855   -0.0316
        0.90     0.1076     0.1177   -0.0101
        1.00     0.1030     0.1030   +0.0000
    peak gap at fraction 0.10 (-0.1965)

==========================================================================

### Seed 2 raw log

  curves: 400 units, 3098s (7.7s/unit), at curves:resnet18:2:fog:5:random_lowres_32:black

============================================================
Completed 510 cells, skipped 0 cached, failed 0 cells
Elapsed time: 9065.7s
manifest written to /kaggle/working/cache/run_manifest.jsonl
============================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
eval images 1000
imputation  mean, blur, uniform_noise, black
explainers  integrated_gradients, grad_cam, random, integrated_gradients_rolled, grad_cam_rolled, random_lowres_2, random_lowres_4, random_lowres_8, random_lowres_16, random_lowres_32

========================================================================
CALIBRATION  (10 cells, n=1000 images)
========================================================================
 sev   accuracy     brier    ece_em    ece_db      aurc
   0     0.8620    0.2047    0.0566    0.0797    0.0264
   1     0.8370    0.2469    0.0674    0.0925    0.0395
   3     0.7273    0.4092    0.1281    0.1441    0.1181
   5     0.5600    0.6764    0.2391    0.2589    0.2707

========================================================================
FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)
========================================================================

  imputation: black
    grad_cam               mean -0.05862   2/10 cells above 0  
    grad_cam_rolled        mean -0.15544   1/10 cells above 0  
    integrated_gradients   mean -0.03902   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients_rolled mean -0.03574   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00000   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.03164   2/10 cells above 0  
    random_lowres_2        mean -0.19694   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean +0.00476   3/10 cells above 0  
    random_lowres_4        mean -0.14463   1/10 cells above 0  
    random_lowres_8        mean -0.08048   2/10 cells above 0  

  imputation: blur
    grad_cam               mean +0.12671   10/10 cells above 0  
    grad_cam_rolled        mean +0.02995   7/10 cells above 0  
    integrated_gradients   mean +0.06888   10/10 cells above 0  
    integrated_gradients_rolled mean -0.00196   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean +0.00003   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean +0.00664   3/10 cells above 0  
    random_lowres_2        mean +0.03896   9/10 cells above 0  
    random_lowres_32       mean +0.01391   8/10 cells above 0  
    random_lowres_4        mean +0.03577   9/10 cells above 0  
    random_lowres_8        mean +0.02057   8/10 cells above 0  

  imputation: mean
    grad_cam               mean -0.03106   2/10 cells above 0  
    grad_cam_rolled        mean -0.14660   1/10 cells above 0  
    integrated_gradients   mean +0.02684   9/10 cells above 0  
    integrated_gradients_rolled mean -0.05072   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00002   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.08051   1/10 cells above 0  
    random_lowres_2        mean -0.15748   1/10 cells above 0  
    random_lowres_32       mean -0.00025   2/10 cells above 0  
    random_lowres_4        mean -0.12443   2/10 cells above 0  
    random_lowres_8        mean -0.10051   2/10 cells above 0  

  imputation: uniform_noise
    grad_cam               mean -0.03637   0/10 cells above 0  NO CELL BEATS RANDOM
    grad_cam_rolled        mean -0.16385   0/10 cells above 0  NO CELL BEATS RANDOM
    integrated_gradients   mean +0.00664   5/10 cells above 0  
    integrated_gradients_rolled mean -0.04529   0/10 cells above 0  NO CELL BEATS RANDOM
    random                 mean -0.00001   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_16       mean -0.07237   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_2        mean -0.17998   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_32       mean -0.00007   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_4        mean -0.15032   0/10 cells above 0  NO CELL BEATS RANDOM
    random_lowres_8        mean -0.11807   0/10 cells above 0  NO CELL BEATS RANDOM

imputation      explainer               sev   faithfulness   half_width
black           grad_cam                  0       -0.07692      0.02089
black           grad_cam                  1       -0.07073      0.02069
black           grad_cam                  3       -0.03720      0.02198
black           grad_cam                  5       -0.06182      0.02185
black           grad_cam_rolled           0       -0.20392      0.02070
black           grad_cam_rolled           1       -0.18745      0.02070
black           grad_cam_rolled           3       -0.13246      0.02218
black           grad_cam_rolled           5       -0.13024      0.02164
black           integrated_gradients      0       -0.04775      0.00665
black           integrated_gradients      1       -0.04731      0.00693
black           integrated_gradients      3       -0.03937      0.00681
black           integrated_gradients      5       -0.02748      0.00621
black           integrated_gradients_rolled    0       -0.04970      0.00643
black           integrated_gradients_rolled    1       -0.04351      0.00626
black           integrated_gradients_rolled    3       -0.03396      0.00663
black           integrated_gradients_rolled    5       -0.02509      0.00595
black           random                    0       -0.00002      0.00011
black           random                    1       -0.00002      0.00011
black           random                    3       +0.00001      0.00010
black           random                    5       +0.00002      0.00012
black           random_lowres_16          0       -0.05742      0.01239
black           random_lowres_16          1       -0.04942      0.01256
black           random_lowres_16          3       -0.02005      0.01324
black           random_lowres_16          5       -0.01686      0.01213
black           random_lowres_2           0       -0.24497      0.02074
black           random_lowres_2           1       -0.23334      0.02082
black           random_lowres_2           3       -0.17572      0.02177
black           random_lowres_2           5       -0.16576      0.02094
black           random_lowres_32          0       +0.00335      0.00574
black           random_lowres_32          1       +0.00378      0.00569
black           random_lowres_32          3       +0.00665      0.00551
black           random_lowres_32          5       +0.00433      0.00521
black           random_lowres_4           0       -0.18965      0.02102
black           random_lowres_4           1       -0.17655      0.02064
black           random_lowres_4           3       -0.12194      0.02135
black           random_lowres_4           5       -0.12041      0.02074
black           random_lowres_8           0       -0.12310      0.01794
black           random_lowres_8           1       -0.10945      0.01839
black           random_lowres_8           3       -0.06088      0.01877
black           random_lowres_8           5       -0.05689      0.01738
blur            grad_cam                  0       +0.12763      0.01249
blur            grad_cam                  1       +0.12975      0.01300
blur            grad_cam                  3       +0.13637      0.01303
blur            grad_cam                  5       +0.11371      0.01304
blur            grad_cam_rolled           0       +0.02896      0.01138
blur            grad_cam_rolled           1       +0.02845      0.01172
blur            grad_cam_rolled           3       +0.03275      0.01219
blur            grad_cam_rolled           5       +0.02897      0.01207
blur            integrated_gradients      0       +0.06358      0.00920
blur            integrated_gradients      1       +0.06533      0.00934
blur            integrated_gradients      3       +0.07629      0.00998
blur            integrated_gradients      5       +0.06678      0.00939
blur            integrated_gradients_rolled    0       -0.00285      0.00785
blur            integrated_gradients_rolled    1       +0.00081      0.00812
blur            integrated_gradients_rolled    3       -0.00134      0.00852
blur            integrated_gradients_rolled    5       -0.00505      0.00825
blur            random                    0       +0.00002      0.00009
blur            random                    1       -0.00000      0.00011
blur            random                    3       +0.00005      0.00013
blur            random                    5       +0.00003      0.00013
blur            random_lowres_16          0       +0.00607      0.00878
blur            random_lowres_16          1       +0.00667      0.00924
blur            random_lowres_16          3       +0.00482      0.00962
blur            random_lowres_16          5       +0.00863      0.00920
blur            random_lowres_2           0       +0.04078      0.01076
blur            random_lowres_2           1       +0.04009      0.01101
blur            random_lowres_2           3       +0.04041      0.01170
blur            random_lowres_2           5       +0.03577      0.01147
blur            random_lowres_32          0       +0.01441      0.00712
blur            random_lowres_32          1       +0.01747      0.00724
blur            random_lowres_32          3       +0.01376      0.00740
blur            random_lowres_32          5       +0.01033      0.00713
blur            random_lowres_4           0       +0.03556      0.01084
blur            random_lowres_4           1       +0.03493      0.01157
blur            random_lowres_4           3       +0.03726      0.01187
blur            random_lowres_4           5       +0.03521      0.01158
blur            random_lowres_8           0       +0.01676      0.01015
blur            random_lowres_8           1       +0.01533      0.01092
blur            random_lowres_8           3       +0.02372      0.01141
blur            random_lowres_8           5       +0.02394      0.01122
mean            grad_cam                  0       -0.06282      0.01718
mean            grad_cam                  1       -0.05124      0.01702
mean            grad_cam                  3       -0.01677      0.01838
mean            grad_cam                  5       -0.01460      0.01872
mean            grad_cam_rolled           0       -0.20544      0.01956
mean            grad_cam_rolled           1       -0.18711      0.01958
mean            grad_cam_rolled           3       -0.12968      0.02011
mean            grad_cam_rolled           5       -0.10340      0.02015
mean            integrated_gradients      0       +0.02998      0.00740
mean            integrated_gradients      1       +0.03098      0.00717
mean            integrated_gradients      3       +0.03025      0.00753
mean            integrated_gradients      5       +0.01824      0.00702
mean            integrated_gradients_rolled    0       -0.06896      0.00768
mean            integrated_gradients_rolled    1       -0.06155      0.00786
mean            integrated_gradients_rolled    3       -0.04496      0.00803
mean            integrated_gradients_rolled    5       -0.03957      0.00784
mean            random                    0       +0.00001      0.00013
mean            random                    1       -0.00004      0.00013
mean            random                    3       -0.00008      0.00017
mean            random                    5       +0.00004      0.00018
mean            random_lowres_16          0       -0.11999      0.01260
mean            random_lowres_16          1       -0.10713      0.01300
mean            random_lowres_16          3       -0.07030      0.01333
mean            random_lowres_16          5       -0.05095      0.01307
mean            random_lowres_2           0       -0.21256      0.01800
mean            random_lowres_2           1       -0.19791      0.01846
mean            random_lowres_2           3       -0.14165      0.01924
mean            random_lowres_2           5       -0.11452      0.01918
mean            random_lowres_32          0       -0.00474      0.00583
mean            random_lowres_32          1       -0.00249      0.00583
mean            random_lowres_32          3       +0.00174      0.00627
mean            random_lowres_32          5       +0.00149      0.00563
mean            random_lowres_4           0       -0.17749      0.01909
mean            random_lowres_4           1       -0.16343      0.01902
mean            random_lowres_4           3       -0.10821      0.01980
mean            random_lowres_4           5       -0.08396      0.01951
mean            random_lowres_8           0       -0.15153      0.01807
mean            random_lowres_8           1       -0.13855      0.01818
mean            random_lowres_8           3       -0.08663      0.01851
mean            random_lowres_8           5       -0.05935      0.01775
uniform_noise   grad_cam                  0       -0.06311      0.00887
uniform_noise   grad_cam                  1       -0.05260      0.00897
uniform_noise   grad_cam                  3       -0.03205      0.00885
uniform_noise   grad_cam                  5       -0.01554      0.00835
uniform_noise   grad_cam_rolled           0       -0.21514      0.01266
uniform_noise   grad_cam_rolled           1       -0.19927      0.01315
uniform_noise   grad_cam_rolled           3       -0.15557      0.01354
uniform_noise   grad_cam_rolled           5       -0.11959      0.01261
uniform_noise   integrated_gradients      0       +0.00935      0.00514
uniform_noise   integrated_gradients      1       +0.01138      0.00504
uniform_noise   integrated_gradients      3       +0.00507      0.00527
uniform_noise   integrated_gradients      5       +0.00256      0.00521
uniform_noise   integrated_gradients_rolled    0       -0.05966      0.00630
uniform_noise   integrated_gradients_rolled    1       -0.05347      0.00607
uniform_noise   integrated_gradients_rolled    3       -0.04145      0.00629
uniform_noise   integrated_gradients_rolled    5       -0.03617      0.00588
uniform_noise   random                    0       +0.00001      0.00013
uniform_noise   random                    1       -0.00003      0.00012
uniform_noise   random                    3       -0.00002      0.00011
uniform_noise   random                    5       +0.00000      0.00011
uniform_noise   random_lowres_16          0       -0.09656      0.00784
uniform_noise   random_lowres_16          1       -0.08876      0.00781
uniform_noise   random_lowres_16          3       -0.06768      0.00774
uniform_noise   random_lowres_16          5       -0.05261      0.00725
uniform_noise   random_lowres_2           0       -0.22722      0.01236
uniform_noise   random_lowres_2           1       -0.21578      0.01248
uniform_noise   random_lowres_2           3       -0.16993      0.01269
uniform_noise   random_lowres_2           5       -0.13848      0.01155
uniform_noise   random_lowres_32          0       +0.00067      0.00370
uniform_noise   random_lowres_32          1       -0.00007      0.00375
uniform_noise   random_lowres_32          3       +0.00005      0.00386
uniform_noise   random_lowres_32          5       -0.00042      0.00378
uniform_noise   random_lowres_4           0       -0.19542      0.01193
uniform_noise   random_lowres_4           1       -0.18287      0.01215
uniform_noise   random_lowres_4           3       -0.14224      0.01230
uniform_noise   random_lowres_4           5       -0.11081      0.01148
uniform_noise   random_lowres_8           0       -0.15774      0.01065
uniform_noise   random_lowres_8           1       -0.14781      0.01060
uniform_noise   random_lowres_8           3       -0.11073      0.01060
uniform_noise   random_lowres_8           5       -0.08246      0.01013

========================================================================
GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid
                  lattice, upsampled as Grad-CAM is. No model input.
========================================================================
imputation       grid   faithfulness   half_width
black               2       -0.19694      0.02113
black               4       -0.14463      0.02092
black               8       -0.08048      0.01815
black              16       -0.03164      0.01262
black              32       +0.00476      0.00550
blur                2       +0.03896      0.01133
blur                4       +0.03577      0.01159
blur                8       +0.02057      0.01108
blur               16       +0.00664      0.00930
blur               32       +0.01391      0.00725
mean                2       -0.15748      0.01886
mean                4       -0.12443      0.01941
mean                8       -0.10051      0.01814
mean               16       -0.08051      0.01308
mean               32       -0.00025      0.00590
uniform_noise       2       -0.17998      0.01225
uniform_noise       4       -0.15032      0.01197
uniform_noise       8       -0.11807      0.01046
uniform_noise      16       -0.07237      0.00763
uniform_noise      32       -0.00007      0.00379

Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on this
curve at grid=4, its score is its resolution and not its content.

========================================================================
GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the
                                same mask geometry; >0 beats it
========================================================================
imputation      explainer                vs                   kind          matched     half_w  verdict
black           grad_cam                 grad_cam_rolled      alignment    +0.09682    0.01174  10/10 cells beat it
black           grad_cam                 random_lowres_4      geometry     +0.08602    0.01134  10/10 cells beat it
black           integrated_gradients     integrated_gradients_rolled alignment    -0.00328    0.00594  NO CELL BEATS ITS MATCHED CONTROL
blur            grad_cam                 grad_cam_rolled      alignment    +0.09676    0.01133  10/10 cells beat it
blur            grad_cam                 random_lowres_4      geometry     +0.09094    0.01037  10/10 cells beat it
blur            integrated_gradients     integrated_gradients_rolled alignment    +0.07084    0.00975  10/10 cells beat it
mean            grad_cam                 grad_cam_rolled      alignment    +0.11554    0.01210  10/10 cells beat it
mean            grad_cam                 random_lowres_4      geometry     +0.09337    0.01172  10/10 cells beat it
mean            integrated_gradients     integrated_gradients_rolled alignment    +0.07756    0.00810  10/10 cells beat it
uniform_noise   grad_cam                 grad_cam_rolled      alignment    +0.12748    0.01205  10/10 cells beat it
uniform_noise   grad_cam                 random_lowres_4      geometry     +0.11395    0.01118  10/10 cells beat it
uniform_noise   integrated_gradients     integrated_gradients_rolled alignment    +0.05194    0.00651  10/10 cells beat it

The pixel-i.i.d. control cancels exactly in this difference: both terms
use the same control_seed over the same images, so what is left is the
explainer scored against its own geometry.

========================================================================
P3 GATE (RAW)   half-width < 10% of the clean-to-severity-5 change
                on `faithfulness`: the explainer against a pixel-iid
                control. The validity run showed that quantity tracks
                mask geometry -- a model-free blob scores well below
                zero on it -- so this table is reported, not read.
========================================================================
imputation      explainer                   family               change     half_w     ratio  verdict
black           grad_cam                    defocus_blur       -0.11739    0.02089     0.178  FAIL
black           grad_cam                    fog                -0.06170    0.02089     0.338  FAIL
black           grad_cam                    gaussian_noise     +0.22439    0.02553     0.114  FAIL
black           grad_cam_rolled             defocus_blur       -0.05044    0.02070     0.410  FAIL
black           grad_cam_rolled             fog                +0.00722    0.02070       n/a  no change
black           grad_cam_rolled             gaussian_noise     +0.26427    0.02532     0.096  PASS
black           integrated_gradients        defocus_blur       +0.00129    0.00665       n/a  no change
black           integrated_gradients        fog                +0.01971    0.00665     0.337  FAIL
black           integrated_gradients        gaussian_noise     +0.03980    0.00665     0.167  FAIL
black           integrated_gradients_rolled defocus_blur       +0.00635    0.00643       n/a  no change
black           integrated_gradients_rolled fog                +0.01899    0.00643     0.339  FAIL
black           integrated_gradients_rolled gaussian_noise     +0.04848    0.00643     0.133  FAIL
black           random                      defocus_blur       +0.00002    0.00011       n/a  no change
black           random                      fog                +0.00004    0.00011       n/a  no change
black           random                      gaussian_noise     +0.00004    0.00016       n/a  no change
black           random_lowres_16            defocus_blur       -0.02475    0.01239     0.501  FAIL
black           random_lowres_16            fog                +0.01989    0.01239     0.623  FAIL
black           random_lowres_16            gaussian_noise     +0.12655    0.01379     0.109  FAIL
black           random_lowres_2             defocus_blur       -0.03210    0.02074     0.646  FAIL
black           random_lowres_2             fog                +0.00154    0.02074       n/a  no change
black           random_lowres_2             gaussian_noise     +0.26819    0.02607     0.097  PASS
black           random_lowres_32            defocus_blur       -0.01052    0.00574     0.546  FAIL
black           random_lowres_32            fog                -0.00558    0.00574       n/a  no change
black           random_lowres_32            gaussian_noise     +0.01904    0.00649     0.341  FAIL
black           random_lowres_4             defocus_blur       -0.05412    0.02102     0.388  FAIL
black           random_lowres_4             fog                +0.01141    0.02102       n/a  no change
black           random_lowres_4             gaussian_noise     +0.25045    0.02429     0.097  PASS
black           random_lowres_8             defocus_blur       -0.03001    0.01794     0.598  FAIL
black           random_lowres_8             fog                +0.02687    0.01794     0.668  FAIL
black           random_lowres_8             gaussian_noise     +0.20175    0.02092     0.104  FAIL
blur            grad_cam                    defocus_blur       -0.08107    0.01249     0.154  FAIL
blur            grad_cam                    fog                -0.01621    0.01249     0.770  FAIL
blur            grad_cam                    gaussian_noise     +0.05555    0.01547     0.279  FAIL
blur            grad_cam_rolled             defocus_blur       -0.03608    0.01138     0.315  FAIL
blur            grad_cam_rolled             fog                -0.02245    0.01138     0.507  FAIL
blur            grad_cam_rolled             gaussian_noise     +0.05858    0.01465     0.250  FAIL
blur            integrated_gradients        defocus_blur       -0.01448    0.00920     0.636  FAIL
blur            integrated_gradients        fog                +0.03200    0.00920     0.287  FAIL
blur            integrated_gradients        gaussian_noise     -0.00792    0.01111       n/a  no change
blur            integrated_gradients_rolled defocus_blur       -0.00687    0.00785       n/a  no change
blur            integrated_gradients_rolled fog                +0.00261    0.00817       n/a  no change
blur            integrated_gradients_rolled gaussian_noise     -0.00236    0.01017       n/a  no change
blur            random                      defocus_blur       +0.00006    0.00011       n/a  no change
blur            random                      fog                +0.00005    0.00015       n/a  no change
blur            random                      gaussian_noise     -0.00007    0.00014       n/a  no change
blur            random_lowres_16            defocus_blur       -0.01425    0.00878     0.616  FAIL
blur            random_lowres_16            fog                -0.00650    0.00878       n/a  no change
blur            random_lowres_16            gaussian_noise     +0.02844    0.01111     0.391  FAIL
blur            random_lowres_2             defocus_blur       -0.04320    0.01076     0.249  FAIL
blur            random_lowres_2             fog                -0.02227    0.01076     0.483  FAIL
blur            random_lowres_2             gaussian_noise     +0.05044    0.01371     0.272  FAIL
blur            random_lowres_32            defocus_blur       -0.01283    0.00712     0.555  FAIL
blur            random_lowres_32            fog                -0.00743    0.00712     0.958  FAIL
blur            random_lowres_32            gaussian_noise     +0.00804    0.00915       n/a  no change
blur            random_lowres_4             defocus_blur       -0.04310    0.01084     0.252  FAIL
blur            random_lowres_4             fog                -0.01713    0.01084     0.633  FAIL
blur            random_lowres_4             gaussian_noise     +0.05919    0.01447     0.244  FAIL
blur            random_lowres_8             defocus_blur       -0.02245    0.01015     0.452  FAIL
blur            random_lowres_8             fog                -0.00519    0.01068       n/a  no change
blur            random_lowres_8             gaussian_noise     +0.04916    0.01383     0.281  FAIL
mean            grad_cam                    defocus_blur       -0.06533    0.01718     0.263  FAIL
mean            grad_cam                    fog                -0.01894    0.01718     0.907  FAIL
mean            grad_cam                    gaussian_noise     +0.22894    0.02344     0.102  FAIL
mean            grad_cam_rolled             defocus_blur       -0.01494    0.01956       n/a  no change
mean            grad_cam_rolled             fog                +0.02873    0.01956     0.681  FAIL
mean            grad_cam_rolled             gaussian_noise     +0.29234    0.02613     0.089  PASS
mean            integrated_gradients        defocus_blur       -0.03555    0.00740     0.208  FAIL
mean            integrated_gradients        fog                -0.01797    0.00740     0.412  FAIL
mean            integrated_gradients        gaussian_noise     +0.01830    0.00769     0.420  FAIL
mean            integrated_gradients_rolled defocus_blur       +0.00938    0.00782     0.833  FAIL
mean            integrated_gradients_rolled fog                +0.01639    0.00788     0.481  FAIL
mean            integrated_gradients_rolled gaussian_noise     +0.06237    0.00783     0.126  FAIL
mean            random                      defocus_blur       +0.00006    0.00015       n/a  no change
mean            random                      fog                +0.00010    0.00024       n/a  no change
mean            random                      gaussian_noise     -0.00006    0.00015       n/a  no change
mean            random_lowres_16            defocus_blur       +0.01065    0.01260       n/a  no change
mean            random_lowres_16            fog                +0.03900    0.01260     0.323  FAIL
mean            random_lowres_16            gaussian_noise     +0.15747    0.01557     0.099  PASS
mean            random_lowres_2             defocus_blur       -0.01089    0.01800       n/a  no change
mean            random_lowres_2             fog                +0.01910    0.01800     0.942  FAIL
mean            random_lowres_2             gaussian_noise     +0.28592    0.02485     0.087  PASS
mean            random_lowres_32            defocus_blur       -0.00575    0.00583       n/a  no change
mean            random_lowres_32            fog                +0.00897    0.00583     0.650  FAIL
mean            random_lowres_32            gaussian_noise     +0.01547    0.00610     0.395  FAIL
mean            random_lowres_4             defocus_blur       -0.02594    0.01909     0.736  FAIL
mean            random_lowres_4             fog                +0.02354    0.01909     0.811  FAIL
mean            random_lowres_4             gaussian_noise     +0.28298    0.02501     0.088  PASS
mean            random_lowres_8             defocus_blur       +0.00054    0.01807       n/a  no change
mean            random_lowres_8             fog                +0.02892    0.01807     0.625  FAIL
mean            random_lowres_8             gaussian_noise     +0.24707    0.02291     0.093  PASS
uniform_noise   grad_cam                    defocus_blur       +0.03125    0.00887     0.284  FAIL
uniform_noise   grad_cam                    fog                +0.03929    0.00887     0.226  FAIL
uniform_noise   grad_cam                    gaussian_noise     +0.07215    0.00950     0.132  FAIL
uniform_noise   grad_cam_rolled             defocus_blur       +0.05385    0.01266     0.235  FAIL
uniform_noise   grad_cam_rolled             fog                +0.06987    0.01266     0.181  FAIL
uniform_noise   grad_cam_rolled             gaussian_noise     +0.16291    0.01437     0.088  PASS
uniform_noise   integrated_gradients        defocus_blur       -0.02057    0.00546     0.265  FAIL
uniform_noise   integrated_gradients        fog                -0.01294    0.00542     0.419  FAIL
uniform_noise   integrated_gradients        gaussian_noise     +0.01315    0.00514     0.391  FAIL
uniform_noise   integrated_gradients_rolled defocus_blur       +0.01286    0.00630     0.490  FAIL
uniform_noise   integrated_gradients_rolled fog                +0.01333    0.00630     0.473  FAIL
uniform_noise   integrated_gradients_rolled gaussian_noise     +0.04429    0.00630     0.142  FAIL
uniform_noise   random                      defocus_blur       -0.00005    0.00013       n/a  no change
uniform_noise   random                      fog                -0.00000    0.00013       n/a  no change
uniform_noise   random                      gaussian_noise     +0.00004    0.00013       n/a  no change
uniform_noise   random_lowres_16            defocus_blur       +0.02323    0.00784     0.338  FAIL
uniform_noise   random_lowres_16            fog                +0.04044    0.00784     0.194  FAIL
uniform_noise   random_lowres_16            gaussian_noise     +0.06818    0.00784     0.115  FAIL
uniform_noise   random_lowres_2             defocus_blur       +0.05273    0.01236     0.234  FAIL
uniform_noise   random_lowres_2             fog                +0.05210    0.01236     0.237  FAIL
uniform_noise   random_lowres_2             gaussian_noise     +0.16139    0.01363     0.084  PASS
uniform_noise   random_lowres_32            defocus_blur       -0.00263    0.00379       n/a  no change
uniform_noise   random_lowres_32            fog                +0.00048    0.00370       n/a  no change
uniform_noise   random_lowres_32            gaussian_noise     -0.00113    0.00392       n/a  no change
uniform_noise   random_lowres_4             defocus_blur       +0.04780    0.01193     0.250  FAIL
uniform_noise   random_lowres_4             fog                +0.06527    0.01193     0.183  FAIL
uniform_noise   random_lowres_4             gaussian_noise     +0.14075    0.01323     0.094  PASS
uniform_noise   random_lowres_8             defocus_blur       +0.04131    0.01065     0.258  FAIL
uniform_noise   random_lowres_8             fog                +0.06589    0.01065     0.162  FAIL
uniform_noise   random_lowres_8             gaussian_noise     +0.11865    0.01090     0.092  PASS

GATE (RAW): 12/87 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================

========================================================================
P3 GATE (GEOMETRY-MATCHED)
   half-width < 10% of the clean-to-severity-5 change in `matched`:
   the explainer minus a control of its OWN mask geometry.
   THIS is the gate the 1000-vs-2000 decision reads. The raw
   gate above gates a quantity the validity run showed
   confounded, so neither its passes nor its failures bear on
   that decision.
========================================================================
imputation      explainer vs reference                              family               change     half_w     ratio  verdict
black           grad_cam vs grad_cam_rolled                         defocus_blur       -0.06695    0.01219     0.182  FAIL
black           grad_cam vs grad_cam_rolled                         fog                -0.06892    0.01219     0.177  FAIL
black           grad_cam vs grad_cam_rolled                         gaussian_noise     -0.03988    0.01219     0.306  FAIL
black           grad_cam vs random_lowres_4                         defocus_blur       -0.06327    0.01192     0.188  FAIL
black           grad_cam vs random_lowres_4                         fog                -0.07311    0.01192     0.163  FAIL
black           grad_cam vs random_lowres_4                         gaussian_noise     -0.02606    0.01192     0.457  FAIL
black           integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.00506    0.00630       n/a  no change
black           integrated_gradients vs integrated_gradients_rolled fog                +0.00072    0.00630       n/a  no change
black           integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.00868    0.00637     0.734  FAIL
blur            grad_cam vs grad_cam_rolled                         defocus_blur       -0.04500    0.01130     0.251  FAIL
blur            grad_cam vs grad_cam_rolled                         fog                +0.00624    0.01130       n/a  no change
blur            grad_cam vs grad_cam_rolled                         gaussian_noise     -0.00303    0.01273       n/a  no change
blur            grad_cam vs random_lowres_4                         defocus_blur       -0.03798    0.01007     0.265  FAIL
blur            grad_cam vs random_lowres_4                         fog                +0.00092    0.01015       n/a  no change
blur            grad_cam vs random_lowres_4                         gaussian_noise     -0.00364    0.01180       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.00761    0.00999       n/a  no change
blur            integrated_gradients vs integrated_gradients_rolled fog                +0.02939    0.00999     0.340  FAIL
blur            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.00556    0.01086       n/a  no change
mean            grad_cam vs grad_cam_rolled                         defocus_blur       -0.05039    0.01267     0.251  FAIL
mean            grad_cam vs grad_cam_rolled                         fog                -0.04767    0.01234     0.259  FAIL
mean            grad_cam vs grad_cam_rolled                         gaussian_noise     -0.06340    0.01234     0.195  FAIL
mean            grad_cam vs random_lowres_4                         defocus_blur       -0.03939    0.01269     0.322  FAIL
mean            grad_cam vs random_lowres_4                         fog                -0.04248    0.01207     0.284  FAIL
mean            grad_cam vs random_lowres_4                         gaussian_noise     -0.05404    0.01207     0.223  FAIL
mean            integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.04493    0.00854     0.190  FAIL
mean            integrated_gradients vs integrated_gradients_rolled fog                -0.03436    0.00854     0.249  FAIL
mean            integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.04407    0.00854     0.194  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         defocus_blur       -0.02260    0.01214     0.537  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         fog                -0.03058    0.01214     0.397  FAIL
uniform_noise   grad_cam vs grad_cam_rolled                         gaussian_noise     -0.09076    0.01214     0.134  FAIL
uniform_noise   grad_cam vs random_lowres_4                         defocus_blur       -0.01655    0.01176     0.710  FAIL
uniform_noise   grad_cam vs random_lowres_4                         fog                -0.02598    0.01176     0.453  FAIL
uniform_noise   grad_cam vs random_lowres_4                         gaussian_noise     -0.06860    0.01176     0.171  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled defocus_blur       -0.03343    0.00670     0.201  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled fog                -0.02627    0.00664     0.253  FAIL
uniform_noise   integrated_gradients vs integrated_gradients_rolled gaussian_noise     -0.03114    0.00664     0.213  FAIL

GATE (GEOMETRY-MATCHED): 0/28 resolvable pairs pass.
Per the pre-registered failure branch, a failing gate means changing the faithfulness metric or pivoting to a measurement-validity paper. It does not mean raising n_eval_images: the half-width falls as 1/sqrt(n), so passing by that route costs 100x the compute for a 10x narrowing and is a data-dependent protocol change besides.
========================================================================
config      /kaggle/working/shiftprofile/1/configs/pilot_validity.yaml
cache read  /kaggle/working/cache
cells       10
imputation  mean, blur
pairs       integrated_gradients vs integrated_gradients_rolled, grad_cam vs grad_cam_rolled
roll seed   0

==========================================================================
TEST 1   does displacing the map FURTHER cost more?
         gain = faithfulness(rolled) - faithfulness(explainer)
         gain < 0 means the explainer beats its own displaced copy;
         |gain| growing with distance means its aim is spatially
         specific, not merely a differently shaped mask
==========================================================================
imputation      explainer              displ_px       gain    half_w      n
mean            integrated_gradients       5.73   -0.05617   0.00527   2040
mean            integrated_gradients       9.82   -0.06845   0.00569   2050
mean            integrated_gradients      12.56   -0.07987   0.00549   2100
mean            integrated_gradients      14.78   -0.08287   0.00581   2100
mean            integrated_gradients      17.90   -0.09853   0.00626   2030
mean            grad_cam                   5.73   -0.04722   0.00596   2040
mean            grad_cam                   9.82   -0.08914   0.00801   2050
mean            grad_cam                  12.56   -0.11157   0.00810   2100
mean            grad_cam                  14.78   -0.13065   0.00931   2100
mean            grad_cam                  17.90   -0.19531   0.00982   2030
blur            integrated_gradients       5.73   -0.05501   0.00659   2040
blur            integrated_gradients       9.82   -0.06789   0.00688   2050
blur            integrated_gradients      12.56   -0.06914   0.00652   2100
blur            integrated_gradients      14.78   -0.07967   0.00705   2100
blur            integrated_gradients      17.90   -0.08670   0.00700   2030
blur            grad_cam                   5.73   -0.02960   0.00461   2040
blur            grad_cam                   9.82   -0.07461   0.00726   2050
blur            grad_cam                  12.56   -0.10019   0.00764   2100
blur            grad_cam                  14.78   -0.12302   0.00855   2100
blur            grad_cam                  17.90   -0.15353   0.00989   2030

  blur            grad_cam               -0.02960 -> -0.15353  |swing| +0.12393 vs half-width 0.00989   advantage GROWS with distance (beats its copy)
  blur            integrated_gradients   -0.05501 -> -0.08670  |swing| +0.03168 vs half-width 0.00700   advantage GROWS with distance (beats its copy)
  mean            grad_cam               -0.04722 -> -0.19531  |swing| +0.14809 vs half-width 0.00982   advantage GROWS with distance (beats its copy)
  mean            integrated_gradients   -0.05617 -> -0.09853  |swing| +0.04236 vs half-width 0.00626   advantage GROWS with distance (beats its copy)

==========================================================================
TEST 2   faithfulness against mask geometry, all arms on one table
         centre_distance: mask centroid's distance from frame centre
         spread: how far masked pixels sit from their own centroid
==========================================================================

  imputation: blur   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.295   6.046      +0.12671  
    random_lowres_2                  3.967  16.017      +0.03896  yes
    grad_cam_rolled                  6.170   9.254      +0.02995  
    random_lowres_4                  6.511  10.994      +0.03577  yes
    integrated_gradients             7.951   3.664      +0.06888  
    integrated_gradients_rolled     10.569   5.008      -0.00196  
    random_lowres_8                 10.897   6.229      +0.02057  yes
    random                          12.112   1.594      +0.00003  yes
    random_lowres_32                12.114   1.538      +0.01391  yes
    random_lowres_16                12.120   3.433      +0.00664  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.29 vs 3.97)  residual +0.08775
      grad_cam_rolled              vs random_lowres_4      (spread 6.17 vs 6.51)  residual -0.00583
      integrated_gradients         vs random_lowres_4      (spread 7.95 vs 6.51)  residual +0.03310
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.57 vs 10.90)  residual -0.02253

  imputation: mean   (ordered by mask spread)
    explainer                       spread  centre  faithfulness  model-free
    grad_cam                         3.295   6.046      -0.03106  
    random_lowres_2                  3.967  16.017      -0.15748  yes
    grad_cam_rolled                  6.170   9.254      -0.14660  
    random_lowres_4                  6.511  10.994      -0.12443  yes
    integrated_gradients             7.951   3.664      +0.02684  
    integrated_gradients_rolled     10.569   5.008      -0.05072  
    random_lowres_8                 10.897   6.229      -0.10051  yes
    random                          12.112   1.594      -0.00002  yes
    random_lowres_32                12.114   1.538      -0.00025  yes
    random_lowres_16                12.120   3.433      -0.08051  yes
    residual of each real arm against the model-free arms of nearest spread:
      grad_cam                     vs random_lowres_2      (spread 3.29 vs 3.97)  residual +0.12642
      grad_cam_rolled              vs random_lowres_4      (spread 6.17 vs 6.51)  residual -0.02217
      integrated_gradients         vs random_lowres_4      (spread 7.95 vs 6.51)  residual +0.15127
      integrated_gradients_rolled  vs random_lowres_8      (spread 10.57 vs 10.90)  residual +0.04979

==========================================================================
TEST 3   where in the removal curve the gap lives
         probability kept by the explainer's mask minus by a rolled one
==========================================================================

  blur / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8587     0.8587   +0.0000
        0.05     0.8154     0.8409   -0.0255
        0.10     0.7651     0.8230   -0.0579
        0.20     0.6835     0.7842   -0.1007
        0.30     0.6145     0.7429   -0.1284
        0.50     0.5079     0.6610   -0.1531
        0.70     0.4555     0.5774   -0.1219
        0.90     0.4375     0.4829   -0.0454
        1.00     0.4346     0.4346   +0.0000
    peak gap at fraction 0.50 (-0.1531)

  blur / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8587     0.8587   +0.0000
        0.05     0.7868     0.8455   -0.0588
        0.10     0.7519     0.8315   -0.0796
        0.20     0.7044     0.7999   -0.0956
        0.30     0.6663     0.7684   -0.1021
        0.50     0.6102     0.7022   -0.0920
        0.70     0.5619     0.6309   -0.0690
        0.90     0.4953     0.5276   -0.0323
        1.00     0.4346     0.4346   +0.0000
    peak gap at fraction 0.30 (-0.1021)

  mean / grad_cam
    fraction  explainer     rolled       gap
        0.00     0.8587     0.8587   +0.0000
        0.05     0.6974     0.7807   -0.0833
        0.10     0.5783     0.7246   -0.1463
        0.20     0.4067     0.6205   -0.2138
        0.30     0.3090     0.5231   -0.2141
        0.50     0.1965     0.3436   -0.1471
        0.70     0.1463     0.2191   -0.0728
        0.90     0.1205     0.1399   -0.0195
        1.00     0.0924     0.0924   +0.0000
    peak gap at fraction 0.30 (-0.2141)

  mean / integrated_gradients
    fraction  explainer     rolled       gap
        0.00     0.8587     0.8587   +0.0000
        0.05     0.5324     0.7110   -0.1786
        0.10     0.4069     0.6071   -0.2003
        0.20     0.2631     0.4451   -0.1820
        0.30     0.2023     0.3271   -0.1248
        0.50     0.1633     0.2137   -0.0504
        0.70     0.1539     0.1794   -0.0255
        0.90     0.1118     0.1220   -0.0102
        1.00     0.0924     0.0924   +0.0000
    peak gap at fraction 0.10 (-0.2003)

==========================================================================
