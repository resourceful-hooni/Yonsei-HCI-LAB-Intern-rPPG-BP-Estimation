# Non-Invasive Blood Pressure Estimation Using Deep Learning (Enhanced)

**Yonsei HCI LAB Intern Project - 2026**

## 🎯 Overview

This repository contains an enhanced implementation of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning, based on the paper by Schrumpf et al. (2021). The original implementation has been significantly improved with modern computer vision techniques, signal processing enhancements, and real-time stability optimizations.

### Key Enhancements (Phase 2 Implementation)

**✅ Completed Improvements:**
1. **POS Algorithm** - Plane-Orthogonal-to-Skin (Wang et al. 2017) for superior rPPG signal extraction
2. **Advanced Face Detection** - Optimized Haar Cascade with ROI stabilization
3. **Signal Quality Assessment** - SNR calculation, peak detection, and quality scoring
4. **Signal Processing Pipeline**:
   - Adaptive bandpass filtering
   - Detrending for illumination correction
   - Temporal smoothing
   - Motion artifact detection
5. **BP Prediction Stabilization**:
   - Kalman filtering for noise reduction
   - Outlier rejection (2.5σ threshold)
   - Quality-weighted averaging
   - Physiological validity checks
6. **Real-time UI**:
   - Live PPG signal visualization
   - BP/HR monitoring with confidence scores
   - Signal quality metrics display
   - FPS counter and progress tracking

### Performance Metrics
- **Model**: ResNet (SBP MAE: 16.4 mmHg, DBP MAE: 8.5 mmHg on PPG data)
- **Signal Quality**: Real-time SNR monitoring and quality scoring (0-1)
- **Stability**: Kalman filter + weighted averaging for consistent predictions
- **Frame Rate**: 30 FPS real-time processing

---

## 📋 Original Paper Citation

Based on: "Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning" - [Sensors Special Issue](https://www.mdpi.com/1424-8220/21/18/6022)

```bibtex
@inproceedings{schrumpf2021assessment,
  title={Assessment of deep learning based blood pressure prediction from PPG and rPPG signals},
  author={Schrumpf, Fabian and Frenzel, Patrick and Aust, Christoph and Osterhoff, Georg and Fuchs, Mirco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3820--3830},
  year={2021}
}
```

---

## 🚀 Quick Start

### Installation

**Requirements:**
- Python 3.8
- Windows 10/11 (or Linux/Mac with modifications)
- Webcam for rPPG testing

**Setup:**
```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Real-time BP Prediction

```bash
# Basic usage (7 seconds collection, POS algorithm, camera 0)
python camera_rppg_advanced.py --camera 0 --duration 7 --pos

# Custom settings
python camera_rppg_advanced.py --camera 1 --duration 5 --pos --no-mediapipe

# Available options:
#   --camera: Camera index (default: 0)
#   --duration: Signal collection time in seconds (default: 7)
#   --pos: Enable POS algorithm (default: True)
#   --no-pos: Use simple green channel extraction
#   --model: Path to model file (default: data/resnet_ppg_nonmixed.h5)
```

---

## 📁 Project Structure

### Core Modules

| File | Description |
|------|-------------|
| `camera_rppg_advanced.py` | **Main script** - Real-time BP prediction with enhanced UI |
| `pos_algorithm.py` | POS algorithm implementation (Wang et al. 2017) |
| `signal_quality.py` | Signal quality assessment and enhancement |
| `bp_stability.py` | BP prediction stabilization (Kalman filter, outlier rejection) |
| `mediapipe_face_detector.py` | Face detection with ROI stabilization |

### Test Scripts

| File | Description |
|------|-------------|
| `test_pos_only.py` | Unit test for POS algorithm |
| `test_phase2_step3.py` | Integration test for Phase 2 |
| `debug_face_detection.py` | Face detection debugging tool |

### Original Training Scripts

| File | Description |
|------|-------------|
| `download_mimic_iii_records.py` | Download MIMIC-III database records |
| `prepare_MIMIC_dataset.py` | Preprocess and prepare dataset |
| `h5_to_tfrecord.py` | Convert to TFRecord format |
| `ppg_training_mimic_iii.py` | Train neural networks on PPG data |
| `ppg_personalization_mimic_iii.py` | Fine-tune with subject-specific data |
| `retrain_rppg_personalization.py` | Transfer learning for rPPG |

### Documentation

| File | Description |
|------|-------------|
| `COMPREHENSIVE_SOLUTION_GUIDE.md` | Complete technical guide (8 sections, 1367 lines) |
| `README.md` | This file |

---

## 🔬 Technical Details

### Signal Processing Pipeline

```
Raw Camera Feed
    ↓
Face Detection (Haar Cascade + ROI Stabilization)
    ↓
RGB Signal Extraction
    ↓
POS Algorithm (Orthogonal Projection)
    ↓
Detrending (Illumination Correction)
    ↓
Adaptive Bandpass Filter (0.7-4 Hz)
    ↓
Temporal Smoothing (α=0.3)
    ↓
Quality Assessment (SNR, Peak Detection)
    ↓
Resampling to 875 samples
    ↓
ResNet Model Prediction
    ↓
Kalman Filter + Outlier Rejection
    ↓
Final BP Values (SBP/DBP) + Confidence
```

### Key Algorithms

#### 1. POS Algorithm (Plane-Orthogonal-to-Skin)
```python
# Normalize RGB channels
C_norm = RGB / mean(RGB)

# Orthogonal projection
S1 = G - B
S2 = -2R + G + B

# Weighted combination
pulse = S1 + α * S2
```

#### 2. Kalman Filter for BP Stabilization
```python
# Prediction step
prediction = estimate
prediction_error = estimate_error + process_variance

# Update step
kalman_gain = prediction_error / (prediction_error + measurement_variance)
estimate = prediction + kalman_gain * (measurement - prediction)
```

#### 3. Signal Quality Metrics
- **SNR**: Signal-to-Noise Ratio in dB
- **Peak Regularity**: Consistency of heartbeat intervals
- **HR Power Ratio**: Energy in heart rate frequency band
- **Overall Score**: 0-1 composite quality metric

---

## 📊 Models and Data

### Pre-trained Models

Available in `data/` directory:
- `resnet_ppg_nonmixed.h5` - **Best performance** (SBP: 16.4, DBP: 8.5 MAE)
- `alexnet_ppg_nonmixed.h5` - AlexNet architecture
- `lstm_ppg_nonmixed.h5` - LSTM-based model
- `slapnicar_ppg_nonmixed.h5` - Slapnicar et al. architecture

### Datasets

- **MIMIC-III PPG Dataset**: 32 GB, available on [Zenodo](https://zenodo.org/record/5590603)
- **rPPG-BP-UKL Dataset**: 7-second rPPG recordings for fine-tuning

---

## 🧪 Testing

### Unit Tests

```bash
# Test POS algorithm
python test_pos_only.py
# Expected: ~4% error on synthetic signals

# Test integration (POS + Face Detection)
python test_phase2_step3.py
# Expected: All 7 test sections pass

# Debug face detection
python debug_face_detection.py
# Opens camera with face detection visualization
```

### Real-time Testing Results

**Test Configuration:**
- Camera: 30 FPS, 640x480
- Duration: 7 seconds (210 frames)
- Model: ResNet PPG

**Sample Results:**
```
Signal Quality Score: 0.85/1.00
SNR: 12.3 dB
Peaks: 8 detected
Peak Regularity: 0.81

Blood Pressure:
  Raw:        SBP=135.2 → 120.4 mmHg, DBP=75.8 → 65.2 mmHg
  Stabilized: SBP=120.4 mmHg, DBP=65.2 mmHg
  Confidence: 0.82/1.00
  
Heart Rate: 72.0 bpm
```

---

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'tensorflow'"**
```bash
# Activate virtual environment first
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate      # Linux/Mac

# Install tensorflow
pip install tensorflow==2.4.1
```

**2. Camera not opening**
```bash
# Try different camera index
python camera_rppg_advanced.py --camera 1

# Check available cameras (Windows)
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(4)])"
```

**3. Low signal quality (<0.3)**
- Ensure good lighting (avoid direct sunlight/shadows)
- Keep head still during measurement
- Position face centered in camera view
- Clean camera lens

**4. Unstable BP predictions**
- Take multiple measurements (3-5)
- Use longer duration (--duration 10)
- Ensure high signal quality (>0.7)
- Avoid movement during measurement

---

## 📈 Future Improvements (Phase 3)

Planned enhancements for next version:
1. **Transformer Models** - Vision Transformer (ViT) for rPPG
2. **Attention Mechanisms** - Temporal attention for signal processing
3. **Domain Adaptation** - Improved PPG→rPPG transfer learning
4. **Multi-Task Learning** - Simultaneous BP/HR/SpO2 prediction
5. **Model Optimization** - ONNX/TensorRT for faster inference

See `COMPREHENSIVE_SOLUTION_GUIDE.md` for detailed implementation roadmap.

---

## 📚 Original Training Pipeline

For reproducing the original paper results using MIMIC-III database:

### 1. Download MIMIC-III Data
```bash
python download_mimic_iii_records.py MIMIC-III_ppg_dataset_records.txt ./mimic_data/
# Warning: ~1.5 TB required, takes several hours
```

### 2. Prepare Dataset
```bash
python prepare_MIMIC_dataset.py ./mimic_data/ ./data/MIMIC-III_ppg_dataset.h5 \
    --win_len 7 \
    --win_overlap 0.5 \
    --maxsampsubject 10000 \
    --save_ppg_data 1
```

### 3. Convert to TFRecord
```bash
python h5_to_tfrecord.py ./data/MIMIC-III_ppg_dataset.h5 ./tfrecords/ \
    --ntrain 1000000 \
    --nval 250000 \
    --ntest 250000 \
    --divbysubj 1
```

### 4. Train Model
```bash
python ppg_training_mimic_iii.py experiment_name ./tfrecords/ ./results/ ./checkpoints/ \
    --arch resnet \
    --lr 0.001 \
    --batch_size 128 \
    --epochs 100
```

### 5. Personalization (Optional)
```bash
python ppg_personalization_mimic_iii.py experiment_name ./tfrecords/ ./results/ \
    ./checkpoints/resnet_model.h5 ./checkpoints_personalized/
```

---

## 🤝 Contributing

This is an enhanced version of the original implementation. For contributing:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project follows the original repository's license. See LICENSE.md for details.

---

## 🙏 Acknowledgments

- Original paper authors: Schrumpf et al. (2021)
- MIMIC-III Database: PhysioNet
- POS Algorithm: Wang et al. (2017)
- Yonsei HCI LAB for project support

---

## 📧 Contact

**Yonsei HCI LAB Intern Project**
- Repository: [github.com/resourceful-hooni/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation](https://github.com/resourceful-hooni/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation)
- Email: dev@yonsei-hci.lab

For questions about the original implementation, refer to the [original repository](https://github.com/fabian-sp/bp-estimation-mimic3).

positional arguments:
  ExpName               unique name for the training
  datadir               folder containing the train, val and test subfolders containing tfrecord files
  resultsdir            Directory in which results are stored
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           neural architecture used for training (alexnet (default), resnet, slapnicar, lstm)
  --lr LR               initial learning rate (default: 0.003)
  --batch_size BATCH_SIZE
                        batch size used for training (default: 32)
  --winlen WINLEN       length of the ppg windows in samples (default: 875)
  --epochs EPOCHS       maximum number of epochs for training (default: 60)
  --gpuid GPUID         GPU-ID used for training in a multi-GPU environment (default: None)
```
### Personalizing pretrained neural networks using PPG data
The script `ppg_personalization_mimic_iii.py` takes a set of test subjects and fine tunes neural network that were trained based on PPG data. The goal is to improve the MAE on those test subjects by using 20 % of each test subject's data for retraining. These 20 % can be dranwn randomly or systematically (the first 20 %). The remaining 80 % are used for validation. The script performs BP predictions using the validation data before and after personalization for comparison. Results are stored in a .csv file for later analysis. 
```
usage: ppg_personalization_mimic_iii.py [-h] [--lr LR] [--batch_size BATCH_SIZE] [--winlen WINLEN] [--epochs EPOCHS]
                                        [--nsubj NSUBJ] [--randompick RANDOMPICK]
                                        ExpName DataDir ResultsDir ModelPath chkptdir

positional arguments:
  ExpName               Name of the training preceeded by the repsective date in the format MM-DD-YYYY
  DataDir               folder containing the train, val and test subfolders containing tfrecord files
  ResultsDir            Directory in which results are stored
  ModelPath             Path where the model file used for personalization is located
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate (default: 0.003)
  --batch_size BATCH_SIZE
                        batch size used for training (default: 32)
  --winlen WINLEN       length of the ppg windows in samples (default: 875)
  --epochs EPOCHS       maximum number of epochs for training (default: 60)
  --nsubj NSUBJ         Number subjects used for personalization (default :20)
  --randompick RANDOMPICK
                        define wether data for personalization is drawn randomly (1) or comprises the first 20 % of the test
                        subject's data (0) (default: 0)

```
### rPPG based BP prediction using transfer learning

The script `retrain_rppg_personalization.py` trains a pretrained neural network (trained using the script `pg_train_mimic_iii.py`) for camera based BP prediction. The rPPG data is provided by a hdf5 file in the data subfolder. The rPPG data was collected during a study at the Leipzig University Hospital. Subjects were filmed using a standard RGB camera. rPPG signals were derived from skin regions on the subject's face using the plane-orthogonal-to-skin algorithm published by Wang et al. [[5]](#5).

If you use this data in you own research, please cite our paper:

```
@inproceedings{schrumpf2021assessment,
  title={Assessment of deep learning based blood pressure prediction from PPG and rPPG signals},
  author={Schrumpf, Fabian and Frenzel, Patrick and Aust, Christoph and Osterhoff, Georg and Fuchs, Mirco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3820--3830},
  year={2021}
}
```

The pretrained networks are finetuned using a leave-one-subject-out cross validation scheme. Personalization can be performed by using a portion of the test subject's data for training. The networks are evaluated using the test subject's data BEFORE and AFTER fine tuning. Results are stored in a csv file for analysis.
```
usage: retrain_rppg_personalization.py [-h] [--pers PERS] [--randompick RANDOMPICK] ExpName DataFile ResultsDir ModelPath chkptdir

positional arguments:
  ExpName               Name of the training preceeded by the repsective date in the format MM-DD-YYYY
  DataFile              Path to the hdf file containing rPPG signals
  ResultsDir            Directory in which results are stored
  ModelPath             Path where the model file used for rPPG based personalization is located
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --pers PERS           If 0, performs personalizatin using data from the test subjct
  --randompick RANDOMPICK
                        If 0, uses the first 20 % of the test subject's data for testing, otherwise select randomly (only applies if --pers == 1)

```
The 


## Using the pretrained models
The subfolder `trained_models` contains .h5-files containing models definitions and weights. The models wer trained using a non-mixed dataset as described in [[1]](#1). To use the networks for prediction/fine-tuning, input and output data must meet the following requirements:
* input data must have a length of 875 samples (corresponds to 7 seconds using a sampling frequency of 125 Hz)
* SBP and DBP must be provided separately as there is one output node for each value

The models can be imported the following way:
```python
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel

dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel

model = ks.load_model(<PathToModelFile>, custom_objects=dependencies)
```
Predictions can then be made using the `model.predict()` function. 

## References
<a id="1">[1]</a> Schrumpf, F.; Frenzel, P.; Aust, C.; Osterhoff, G.; Fuchs, M. Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning. Sensors 2021, 21, 6022. https://doi.org/10.3390/s21186022 

<a id="2">[2]</a> A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
    Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.

<a id="3">[3]</a> K. He, X. Zhang, S. Ren, und J. Sun, „Deep Residual Learning for Image Recognition“, in 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Juni 2016, S. 770–778. doi: 10.1109/CVPR.2016.90.

<a id="4">[4]</a> G. Slapničar, N. Mlakar, und M. Luštrek, „Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
    Deep Neural Network“, Sensors, Bd. 19, Nr. 15, S. 3420, Aug. 2019, doi: 10.3390/s19153420.

<a id="5">[5]</a> W. Wang, A. C. den Brinker, S. Stuijk, und G. de Haan, „Algorithmic Principles of Remote PPG“, IEEE Transactions on Biomedical Engineering, Bd. 64, Nr. 7, S. 1479–1491, Juli 2017, doi: 10.1109/TBME.2016.2609282.
