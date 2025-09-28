# Pneumonia Detection with Deep Learning

This project uses a pre-trained **ResNet-18** model to classify chest X-ray images into two categories: **Normal** or **Pneumonia**. It leverages PyTorch for model training and evaluation, along with `kagglehub` for downloading the dataset programmatically.

---

## Dataset

The dataset used is the **Chest X-ray Pneumonia Dataset** by Paul Timothy Mooney, available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

It contains three subsets:

* `train/` – training images
* `val/` – validation images
* `test/` – test images

The project downloads the latest version automatically to a local directory using `kagglehub`.

---

## Requirements

Python 3.10+ and the following packages:

```bash
torch>=2.8.0
torchvision>=0.15.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Pillow>=9.0.0
tqdm>=4.65.0
kagglehub>=0.1.0
numpy>=1.25.0
```

> **Note:** If you have a CUDA-compatible GPU, ensure you install the GPU version of PyTorch to leverage faster training.

---

## Setup

1. Clone this repository:

```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

* **Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

* **Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

* **Linux / macOS:**

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the main script to download the dataset, train the model, and evaluate:

```bash
python github_folder/main.py
```

* The model will automatically detect if a GPU is available.
* Training progress and evaluation metrics are displayed in the terminal.
* A confusion matrix and classification report are printed after evaluation.
* Predictions from the test set are visualized in a plot.

---

## Outputs

1. **Trained Model:** `pneumonia_detector.pth`
2. **Classification Report & Confusion Matrix:** Printed in terminal
3. **Prediction Visualization:** A matplotlib plot showing predicted vs true labels

---

## Project Structure

```
pneumonia_machine_learning/
│
├─ github_folder/
│   └─ main.py          # Training, evaluation, and visualization script
|   └─ requirements.txt
│
├─ chest_xray/          # Dataset directory (downloaded automatically)
│   ├─ train/
│   ├─ val/
│   └─ test/
│
├─ pneumonia_detector.pth
└─ README.md
```

---

## Notes

* Ensure your GPU drivers and CUDA toolkit are installed if you want to leverage GPU acceleration.
* Adjust `num_epochs`, `batch_size`, and learning rate in `main.py` for better performance or faster experimentation.
* The script uses ResNet-18 as the backbone, but you can easily switch to other models from `torchvision.models`.

---

## License

This project is for educational purposes. Dataset is provided under [Kaggle’s terms](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
