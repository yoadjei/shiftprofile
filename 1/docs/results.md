# Run logs

Two parts, and they must not be mixed. **Part 1** is the VOID runs, kept for
provenance; every faithfulness figure in it is wrong. **Part 2** is the
corrected run under `curves-v3` and is the one to cite.

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
  scripts/mechanism_report.py at commit 77781e8; this log predates the fix.]

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
