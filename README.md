# LGRC-Net

LGRC-Net is a printed circuit board (PCB) defect detector based on local–global feature modeling and residual compensation. This repository provides the model architecture, custom layers, and the CIoU + NWD box regression loss implementation.

## Installation

Run the following commands from the repository root in a Python environment. These commands install a matching CPU build of PyTorch and torchvision, followed by the project dependencies.

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Build the model

```bash
python -c "from ultralytics import YOLO; YOLO('configs/LGRC-Net.yaml', task='detect').info()"
```

This command builds LGRC-Net from the YAML configuration with randomly initialized parameters and prints the model summary.

## Source files

- [`configs/LGRC-Net.yaml`](configs/LGRC-Net.yaml) defines the network architecture.
- [`ultralytics/nn/extra_modules/`](ultralytics/nn/extra_modules/) contains the custom layers.
- [`ultralytics/nn/tasks.py`](ultralytics/nn/tasks.py) registers the layers and builds the model.
- [`ultralytics/utils/loss.py`](ultralytics/utils/loss.py) implements the box regression loss.

## Weights and license

Trained model weights are not included in this repository. See [LICENSE](LICENSE) for the GNU Affero General Public License v3.0.
