# Wi-Fi-Based Fall Detection Using ESP32 CSI and Deep Learning

> A privacy-preserving fall detection system that leverages Wi-Fi Channel State Information (CSI) and Deep Learning to classify fall and non-fall events.

---

## Overview

Falls are a significant safety concern, particularly for elderly individuals and people requiring continuous monitoring. Traditional fall detection systems often rely on wearable sensors or camera-based monitoring, which can introduce challenges related to user convenience and privacy.

This project explores a **contactless and privacy-preserving approach to fall detection** using **Wi-Fi Channel State Information (CSI)** collected through an **ESP32-based Wi-Fi sensing environment**.

Changes in human movement affect Wi-Fi signal propagation. These variations are captured through CSI data and processed using a **Convolutional Neural Network (CNN)** to distinguish between **fall** and **non-fall** activities.

---

## Key Features

- Wi-Fi-based contactless fall detection
- ESP32 Channel State Information (CSI) data
- Deep Learning-based activity classification
- Binary classification of **Fall** and **Non-Fall** events
- CSI amplitude-based feature representation
- CNN baseline model implemented using PyTorch
- Privacy-friendly alternative to camera-based monitoring
- Designed as a foundation for future real-time IoT deployment

---

## System Architecture

```text
Human Activity
      │
      ▼
Wi-Fi Signal Propagation
      │
      ▼
ESP32 CSI Data Collection
      │
      ▼
CSI Data Preprocessing
      │
      ▼
Deep Learning Model
      │
      ▼
┌─────────────────────┐
│  Fall / Non-Fall    │
│    Classification   │
└─────────────────────┘
```

---

## Dataset

This project uses an **ESP32 CSI Human Activity and Fall Detection Dataset** containing Wi-Fi Channel State Information recordings for fall detection and human activity recognition.

### Fall Detection Classes

| Label | Class |
|---|---|
| `0` | Non-Fall |
| `1` | Fall |

The dataset contains:

- **Fall samples**
- **Non-Fall samples**
- CSI amplitude information
- **52 valid LLTF subcarriers**

Each sample is stored in `.h5` format.

### Example CSI Data Shape

```text
(52, 500, 1)
```

Where:

- **52** represents the valid Wi-Fi CSI subcarriers.
- **500** represents the captured time samples.
- **1** represents the CSI amplitude channel.

---

## Technologies Used

### Programming and Frameworks

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Data Processing and Visualization

- h5py
- Matplotlib

### Development Environment

- Google Colab
- Jupyter Notebook

### Hardware Platform

- ESP32
- Wi-Fi Channel State Information (CSI)

---

## Methodology

The project follows the workflow below:

### 1. Data Acquisition

CSI data is obtained from the ESP32-based Wi-Fi CSI dataset. The dataset contains `.h5` files with CSI amplitude information.

### 2. Data Preprocessing

The collected CSI data is processed before model training.

Key preprocessing steps include:

- Loading `.h5` CSI files
- Extracting CSI amplitude data
- Assigning class labels
- Converting samples into numerical arrays
- Preparing tensors for Deep Learning
- Splitting the dataset into training and testing sets

### 3. Deep Learning Model

A **Convolutional Neural Network (CNN)** is used as the baseline model.

The CNN automatically learns relevant patterns from CSI amplitude data associated with human movement and fall events.

### 4. Training

The model is trained using labeled CSI samples from the two classes:

```text
Non-Fall → 0
Fall     → 1
```

### 5. Evaluation

The trained model can be evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Project Structure

```text
wifi-fall-detection/
│
├── notebooks/
│   └── fall_detection.ipynb
│
├── models/
│   └── baseline_cnn.pth
│
├── data/
│   ├── fall/
│   └── nonfall/
│
├── requirements.txt
│
└── README.md
```

> **Note:** Large datasets and trained model files may not be included in the repository.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### Install Dependencies

```bash
pip install torch numpy pandas matplotlib h5py scikit-learn
```

---

## Running the Project

### Using Google Colab

1. Open the project notebook.
2. Connect the required dataset.
3. Install the required dependencies.
4. Execute the preprocessing cells.
5. Train the CNN model.
6. Evaluate the model using the test dataset.

---

## Model Pipeline

```text
CSI Dataset
     │
     ▼
Data Loading
     │
     ▼
CSI Amplitude Extraction
     │
     ▼
Data Preprocessing
     │
     ▼
Train / Test Split
     │
     ▼
CNN Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Fall / Non-Fall Prediction
```

---

## Privacy Advantages

Unlike conventional camera-based monitoring systems, this approach:

- Does not capture images or video.
- Does not require users to wear sensors.
- Supports contactless activity monitoring.
- Provides a more privacy-conscious approach to indoor monitoring.

---

## Future Improvements

The project can be extended in several directions:

- **Real-time CSI collection using physical ESP32 hardware**
- **Real-time fall detection**
- Integration with IoT alert systems
- Emergency notifications for caregivers or family members
- CNN-LSTM or Transformer-based architectures
- Multi-class Human Activity Recognition
- Edge AI deployment
- Web or mobile-based monitoring dashboard

### Proposed Real-Time Architecture

```text
Human Activity
      │
      ▼
ESP32 Wi-Fi CSI Collection
      │
      ▼
Real-Time Data Processing
      │
      ▼
Trained Deep Learning Model
      │
      ▼
Fall Detected?
   ┌──┴──┐
   │     │
  No    Yes
   │     │
   ▼     ▼
Monitor  Send Alert
```

---

## Research Applications

This approach can potentially be applied to:

- Elderly care
- Smart homes
- Assisted living environments
- Healthcare monitoring
- Human Activity Recognition (HAR)
- Privacy-preserving sensing systems

---

## Author

**Jees Thomas Cleetus**  
MCA Student | Deep Learning and AI Enthusiast

---

## Acknowledgements

This project is based on the use of **ESP32 Wi-Fi Channel State Information (CSI)** for Human Activity Recognition and Fall Detection research.

---

## License

This project is intended for **academic and research purposes**.
