# 🚗 ATOS Vehicle Re-Identification (Re-ID) Datasets & Benchmarks

This directory contains configuration, annotations, and setup instructions for evaluating vehicle re-identification models in ATOS v3.5.

> [!IMPORTANT]
> **Market-1501 Exclusion Notice**: Market-1501 is a *person* re-identification dataset and is **strictly excluded** from ATOS vehicle Re-ID pipelines and benchmarks.

---

## 📊 Verified Vehicle Re-ID Datasets

The following datasets are supported by the ATOS Re-ID evaluation suite (`scripts/benchmark_reid.py`):

### 1. VeRi-776
- **Description**: 50,000+ images of 776 vehicle identities captured by 20 cameras in an unconstrained urban traffic environment. Includes camera topology, spatiotemporal timestamps, vehicle category, and color labels.
- **Primary Metrics**: Rank-1, Rank-5, mAP (mean Average Precision).
- **License**: Non-commercial academic research license (Institute of Digital Media, Peking University).
- **Official URL**: [VeRi-776 Homepage](https://vecam.github.io/VeRi/)
- **Target Folder Structure**:
  ```text
  datasets/reid/veri776/
  ├── image_train/
  ├── image_test/
  ├── image_query/
  ├── name_train.txt
  ├── name_test.txt
  └── name_query.txt
  ```

### 2. CityFlow-ReID (AI City Challenge)
- **Description**: 229,147 images of 666 vehicle identities across 40 cameras in 3 US cities. Features multi-camera tracking and long-distance trajectories.
- **Primary Metrics**: Rank-1, Rank-5, mAP across distinct camera topologies.
- **License**: NVIDIA AI City Challenge License.
- **Official URL**: [AI City Challenge](https://www.aicitychallenge.org/)
- **Target Folder Structure**:
  ```text
  datasets/reid/cityflow/
  ├── train/
  ├── test/
  └── query/
  ```

### 3. VehicleID
- **Description**: 221,763 images of 26,267 vehicles captured by multiple daytime surveillance cameras in real-world traffic scenes.
- **License**: Non-commercial research license (Peking University).
- **Target Folder Structure**:
  ```text
  datasets/reid/vehicleid/
  ├── image/
  └── train_test_split/
  ```

---

## ⚙️ How to Setup a Dataset for Benchmarking

1. Request access from the official dataset maintainers via the links above.
2. Extract the downloaded dataset archives into `datasets/reid/<dataset_name>/`.
3. Verify that the dataset files are un-tracked by git (enforced by `.gitignore`).
4. Run the benchmark harness:
   ```bash
   python scripts/benchmark_reid.py --dataset veri776 --dataset-dir datasets/reid/veri776 --model models/reid_vehiclenet.onnx
   ```
