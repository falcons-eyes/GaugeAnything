# ADT ATEK Access/Depth Probe — E-dyn-3a/b/c

날짜: 2026-06-12  
목적: 사용자 액션으로 확보된 ADT access JSON이 실제 동적·미터법 GT 실험으로 이어질 수 있는지 검증.

## 입력

- Access URL files, local only: `/Users/jamesjoo/work/falconoon.com/ADT jsons/*.json`
- 주의: JSON 내부 URL은 signed URL이므로 repo에 커밋하지 않는다.
- Cubercnn geometry sequence: `Apartment_release_golden_skeleton_seq100_10s_sample_M1292`
- EFM depth sequences:
  - `Apartment_release_golden_skeleton_seq100_10s_sample_M1292`
  - `Apartment_release_multiuser_clean_seq119_M1292`
- 다운로드한 ATEK cubercnn shards, ignored dataset dir:
  - `datasets/adt_atek_sample/Apartment_release_golden_skeleton_seq100_10s_sample_M1292_shards-0000.tar`
    - sha1 `5e6acc3ad48cb3727cd8f7b513077282819992ab`
  - `datasets/adt_atek_sample/Apartment_release_golden_skeleton_seq100_10s_sample_M1292_shards-0001.tar`
    - sha1 `9145973f19389ede69ee149c3a3afd9ef64eef86`
- 다운로드한 ATEK EFM depth shards, ignored dataset dir:
  - `datasets/adt_atek_efm/Apartment_release_golden_skeleton_seq100_10s_sample_M1292_efm_shards-0000.tar`
    - sha1 `39d12f56bce6ac90f47da36aaf6cdb5f473def02`
  - `datasets/adt_atek_efm/Apartment_release_multiuser_clean_seq119_M1292_efm_shards-0000.tar`
    - sha1 `f6cfb98b4e827a258d821981ec8f32088f5b77a4`
  - `datasets/adt_atek_efm/Apartment_release_multiuser_clean_seq119_M1292_efm_shards-0001.tar`
    - sha1 `4f9376b08aaaf3e08222e3de2305bdfc524e616b`

## 실행

```bash
python experiments/adt_atek_projection_audit.py
```

산출:

- `experiments/results/adt_atek_projection_audit.json`
- `experiments/results/adt_atek_box_dimension_upper.json`
- `experiments/results/adt_atek_depth_upper.json`
- `experiments/results/adt_atek_depth_upper_zmode.json` (depth-mode diagnostic)
- `experiments/results/adt_atek_depth_roi_ablation.json` (no object-volume gate negative control)
- `docs/assets/adt_atek_projection_audit.png`
- `docs/assets/adt_atek_depth_upper.png`
- `docs/assets/adt_dynamic_multiseq_summary.png`

## 프로토콜

ATEK record의 RGB frame, 3-D oriented bounding boxes, object dimensions, camera/device pose, intrinsics를 사용한다.
각 object의 3-D box corner를 다음 체인으로 RGB camera에 투영하고, released 2-D box와 비교한다.

```text
T_camera_object = inv(T_device_camera) @ inv(T_world_device) @ T_world_object
```

평가 필터:

- visibility >= 0.5
- released 2-D box width/height >= 8 px

## 결과

| 지표 | 값 |
|---|---:|
| frames | 50 |
| evaluated instances | 3264 |
| mean IoU / median IoU | 0.658 / 0.675 |
| p10 IoU | 0.385 |
| pass@IoU 0.50 / 0.75 | 80.4% / 35.5% |
| median center error | 3.86 px |
| camera speed median / p90 / max | 0.094 / 0.215 / 0.313 m/s |

## 판정

ADT access는 실제로 작동한다. ATEK shard만으로도 RGB, frame pose, camera intrinsics, 3-D object
dimensions, 2-D/3-D boxes가 동시에 로드되며, 3-D GT를 이미지로 재투영했을 때 released 2-D box와
중앙 IoU 0.675로 정합된다. 이는 E-dyn-3의 최소 geometry chain 검증이다.

## 추가: E-dyn-3b box-only dimension upper bound

질문: released 2-D box와 ADT GT pose/intrinsics만 있으면, 같은 instance의 여러 프레임에서 3-D 치수를
역추정할 수 있는가?

```bash
python experiments/adt_atek_box_dimension_upper.py
```

프로토콜:

- instance별 constant dimension vector `(x, y, z)`를 최적화
- 입력: released 2-D boxes, GT object pose, GT camera/device pose, intrinsics
- 비교: ADT `object_dimensions`
- 필터: visibility >= 0.5, 2-D box width/height >= 16 px, views >= 5

| 지표 | 값 |
|---|---:|
| fitted instances | 75 |
| axis-median relative error, median / p90 | 25.4% / 93.9% |
| axis-mean relative error, median | 40.1% |
| pass@10% / pass@25% | 17.3% / 49.3% |

판정: box-only inverse dimension은 상당히 ill-posed이다. 특히 depth 방향 dimension이 쉽게 하한으로
붕괴한다. 따라서 ADT에서 "promptable 치수"를 주장하려면 2-D box만으로 우회하면 안 되고, depth,
mask/segmentation, 또는 더 강한 multiview fusion을 넣어야 한다.

## 추가: E-dyn-3c oracle-volume depth fusion

질문: ADT EFM의 RGB-depth와 GT object pose/volume을 oracle gate로 쓰면, 여러 프레임 depth를 object
coordinate로 fuse해서 물체 치수를 복원할 수 있는가?

```bash
python experiments/adt_atek_depth_upper.py --depth-mode ray
```

프로토콜:

- 입력: ATEK `efm` shard, RGB-depth `20×1×240×240`, Fisheye624 intrinsics, frame pose, object pose/dimensions
- object 3-D box를 RGB fisheye로 투영해 후보 ROI를 잡고, depth point를 camera→object coordinate로 변환
- GT object volume 안에 들어오는 점만 oracle mask로 사용
- instance별 여러 frame의 object-coordinate points를 fuse하고 2-98 percentile extent를 치수로 읽음
- `other` category 제외

| 지표 | 값 |
|---|---:|
| sequences / shards | 2 / 3 |
| frames | 480 |
| frame observations | 18,036 |
| fitted instances | 229 |
| axis-median relative error, median / p90 | 8.7% / 36.1% |
| axis-mean relative error, median | 15.8% |
| pass@10% / pass@25% | 56.3% / 79.5% |
| ROI-only negative control | 251 instances / median 316.0% / p90 1367.2% / pass@10% 0.0% |

시퀀스/속도별 분해:

| subset | n | camera speed median | median error | p90 error |
|---|---:|---:|---:|---:|
| golden_skeleton_seq100 | 52 | 0.11 m/s | 8.6% | 44.8% |
| multiuser_clean_seq119 | 177 | 0.61 m/s | 8.9% | 33.7% |
| speed 0.00-0.10 m/s | 10 | 0.07 m/s | 28.0% | 55.0% |
| speed 0.10-0.25 m/s | 43 | 0.11 m/s | 8.3% | 34.0% |
| speed 0.25-0.50 m/s | 16 | 0.47 m/s | 5.3% | 43.4% |
| **speed 0.50+ m/s** | **160** | **0.61 m/s** | **9.1%** | **32.5%** |

보조 진단:

- `depth-mode=ray`: 229 instances, median 8.7%, p90 36.1%
- `depth-mode=z`: 127 instances, median 9.8%, p90 28.8%
- 따라서 EFM depth는 ray-distance 해석이 coverage 면에서 더 유리하고, z-mode는 coverage가 줄지만 p90이 낮은 보수 진단으로 볼 수 있다.

판정: depth/multiview를 쓰면 box-only 25.4%보다 훨씬 좋아진다. 특히 두 번째 `multiuser_clean`
시퀀스는 median camera speed 0.61m/s인데도 median 8.9%이고, 0.5m/s 이상 speed bin 전체도 9.1%다.
이는 "움직이는 에고센트릭 RGB-D에서도 미터법 치수 신호가 보존된다"는 현재 가장 강한 동적 증거다.
동시에 ROI-only depth fusion은 median 316.0%로 붕괴한다. 즉 "그냥 박스 안 depth를 모으면 된다"는
우회는 성립하지 않고, object mask/volume gate가 실제 모델 과제다. 다만 8.7% 결과는 GT volume/pose를
oracle gate로 쓰므로 아직 promptable 성능이 아니다. 다음 단계는 ADT segmentation/depth 또는 SAM3
mask로 oracle gate를 대체하는 것이다.

아직 주장하면 안 되는 것:

- SAM3 promptable object measurement 성능
- ADT 전체 시퀀스 일반화
- mask 기반 치수 정확도

다음 단계:

1. E-dyn-3d: ADT segmentation 또는 SAM3 mask로 oracle GT-volume gate를 대체.
2. E-dyn-3e: prompt `"cup"`, `"bottle"`, `"door"`, `"box"` 등으로 mask/depth fusion을 수행하고 GT 3-D dimensions와 비교.
3. sequence 수를 2개에서 더 늘리고, speed/blur/occlusion bin별 gate failure와 치수 오차를 함께 보고.
