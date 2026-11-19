# LGRC-Net 源码整理版

本工程以 2026-10-24 源码包为基础，使用包内的 Ultralytics 8.3.9。LGRC-Net 主模型在骨干第 6、8 层使用标准 `C3k2`，并保留原有边缘增强、P5 编码、局部细化、三尺度融合及检测头实现。

随包提供模型配置和运行脚本。历史 `exp89` 参数仅供参考，复现实验需使用实际的数据划分、配置和对应权重。

## 1. 包内文件

| 文件或目录 | 用途 |
| --- | --- |
| `configs/LGRC-Net.yaml` | 主模型配置，骨干第 6、8 层使用标准 C3k2 |
| `ultralytics/nn/extra_modules/` | 骨干、局部细化、TSRF、检测头等实际模型实现 |
| `ultralytics/nn/tasks.py` | 模型解析、步长初始化、设备转换与融合支持 |
| `ultralytics/utils/loss.py` | CIoU / NWD 混合回归损失以及分类、DFL 损失 |
| `configs/ablations/` | 9 份消融配置，按当前主模型骨干同步更新 |
| `configs/train_exp89_reference.yaml` | 从 exp89 的 args.yaml 提取的主要训练设置 |
| `configs/pcb_data.example.yaml` | 数据集配置示例，需要核对路径和类别编号 |
| `train.py` | 新整理的训练入口，明确记录初始化方式与损失开关 |
| `val.py` | 验证集 / 测试集评估，另存汇总指标 JSON |
| `benchmark.py` | 模型前向 FPS 测量，支持 FP32、FP16 和融合 |
| `predict.py` | 图片 / 文件夹推理并保存结果 |
| `check_model.py` | 结构、前向、反向、权重重载、融合一致性检查 |
| `verification.json` | 本次 CPU 功能检查结果 |
| `reference/` | 历史训练参数、原始依赖及整理说明 |

## 2. AutoDL 安装

建议解压到一个新目录。以下命令从解压后的工程根目录执行，确保导入的是包内 `ultralytics`。

```bash
cd /root/autodl-tmp/LGRC-Net
conda create -n lgrc python=3.10 -y
conda activate lgrc
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available()); print(ultralytics.__file__)"
```

原上传环境使用 PyTorch 2.3.0+cu121 / torchvision 0.18.0+cu121；上面沿用这组版本。需要支持 CUDA 12.1 的 NVIDIA 驱动。依赖文件是本整理版的兼容范围，原始依赖记录在 `reference/requirements_original_20261018.txt`。

本工程已带基础框架，不需要另外安装最新版 `ultralytics`。示例使用无图形界面的 OpenCV，适合服务器保存图片结果。所有路径应替换成你机器上的实际路径。

## 3. 数据集

优先沿用你原工程中的 `data.yaml`，确保 train / val / test 路径有效，类别编号与训练标签一致。例如：

```yaml
path: /root/autodl-tmp/BJUPCB
train: images/train
val: images/val
test: images/test
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

**上述六类顺序仅为示例，不能据此更改你现有标签的类别编号。** 本次没有恢复到原数据集 YAML，不能凭类别名字确认你当时的编号顺序。

## 4. 训练与损失设置

原始源码快照中，`BboxLoss.nwd_loss=False`，`iou_ratio=0.1`，NWD 常数为 12.8。它只能说明上传源码当时的设置，不能证明历史上每一次训练是否启用了 NWD。原 YAML 注释里的 `NWD03` 也不控制实际损失。

整理版 **默认关闭 NWD，与恢复到的源码开关一致**。若要使用论文讨论的 CIoU 与 NWD 混合损失，必须明确传 `--nwd on`。例如：

```bash
python train.py --data /root/autodl-tmp/BJUPCB/data.yaml --epochs 300 --batch 32 --imgsz 640 --device 0 --seed 0 --nwd on --iou-ratio 0.1 --name LGRC-Net_seed0
```

仅用 CIoU 的对应命令：

```bash
python train.py --data /root/autodl-tmp/BJUPCB/data.yaml --epochs 300 --batch 32 --imgsz 640 --device 0 --seed 0 --nwd off --name LGRC-Net_CIoU_seed0
```

NWD 开启时，框回归部分为：

```text
L_reg = alpha * L_CIoU + (1 - alpha) * L_NWD
alpha = --iou-ratio
L_total = 7.5 * L_reg + 0.5 * L_cls + 1.5 * L_DFL
```

正样本加权、归一化和 NWD 计算方式沿用原实现。NWD 接收的是除以各检测层步长后的框坐标，常数 12.8 对应此处的特征网格单位。改变常数或坐标单位会改变损失，需作为新的实验设置记录。

训练入口默认从 YAML 随机初始化。要加载初始权重，明确传入：

```bash
python train.py --data /root/autodl-tmp/BJUPCB/data.yaml --init-weights /root/autodl-tmp/yolo11n.pt --nwd on --name LGRC-Net_pretrained_seed0
```

框架会按匹配的层和形状加载权重，不能因此假定所有自定义层都来自预训练。原 exp89 的 `pretrained: true` 参数也不足以单独证明当时实际加载了预训练权重。

修改学习率等参数时使用 `--set`：

```bash
python train.py --data /root/autodl-tmp/BJUPCB/data.yaml --nwd on --set lr0=0.005 --set weight_decay=0.0005 --seed 0 --name LGRC-Net_trial
```

每次训练会在结果目录保存 `lgrc_run.json` 和 `code_snapshot/`，记录损失开关、混合系数、初始化方式、训练参数和关键源码哈希。损失配置也保存到新训练的权重中。

训练中断时，可以使用尚未完成全部轮数的 `last.pt` 续训；默认保留该权重中记录的损失设置：

```bash
python train.py --data /root/autodl-tmp/BJUPCB/data.yaml --resume runs/train/LGRC-Net_seed0/weights/last.pt
```

如果历史权重没有保存损失配置，续训时应依据当时实际设置明确传 `--nwd on` 或 `--nwd off`。这份工程不能自动还原未记录的历史开关。

## 5. 测试集评估

```bash
python val.py --weights runs/train/LGRC-Net_seed0/weights/best.pt --data /root/autodl-tmp/BJUPCB/data.yaml --split test --imgsz 640 --batch 16 --device 0
```

结果写入 `runs/test/`，包括 `lgrc_metrics.json`。其中 F1 使用汇总 P、R 计算 `2PR/(P+R)`；它不等同于先计算每一类 F1 再取平均。原始框架验证指标和本整理入口的 F1 分别保留自己的含义。

测试集不应参与选择训练轮数、超参数或候选模型；这些选择使用验证集。

## 6. FPS 测量

FP32、batch=1：

```bash
python benchmark.py --weights runs/train/LGRC-Net_seed0/weights/best.pt --device 0 --imgsz 640 --batch 1 --warmup 20 --runs 200 --output fps_fp32.json
```

FP16、batch=1：

```bash
python benchmark.py --weights runs/train/LGRC-Net_seed0/weights/best.pt --device 0 --imgsz 640 --batch 1 --warmup 20 --runs 200 --half --output fps_fp16.json
```

默认执行 Conv-BN 融合及 DEConv 重参数化。计时包含模型前向和模型内部框解码，排除读取图片、预处理、外部 NMS；使用随机输入，先预热并同步 CUDA。不同模型需使用相同硬件、输入尺寸、精度、batch 和计时范围。FP16 不改变模型本身的理论 GFLOPs。

## 7. 推理、自检与消融

```bash
python predict.py --weights runs/train/LGRC-Net_seed0/weights/best.pt --source /root/autodl-tmp/BJUPCB/images/test --device 0
python check_model.py --all-ablations
```

消融训练通过替换模型 YAML：

```bash
python train.py --model configs/ablations/LGRC-Net-NoTSRF.yaml --data /root/autodl-tmp/BJUPCB/data.yaml --nwd on --seed 0 --name NoTSRF_seed0
```

消融配置中的第 6、8 层按各自设计选用标准 C3k2。引用历史指标时，请核对实际运行对应的模型配置、损失设置和随机种子。配置之间的对应关系见 `reference/RESTORATION.md`。

## 8. 本次验证范围

运行 `python check_model.py --all-ablations` 可检查主模型及 9 份消融配置。主模型检查包括前向推理、损失反向传播、权重保存与重载，以及融合前后的输出一致性。

完整版本的核对记录见 `verification.json`。功能检查使用随机生成的输入，真实数据集的 mAP 和 GPU FPS 需在对应实验环境验证。原有许可证见 `LICENSE`。
