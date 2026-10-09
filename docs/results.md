# Run logs — VOID, kept for provenance

**Every faithfulness, gate, ladder and matched-control number below is wrong.**
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