深度学习框架使用YOLOv8训练无人机视角下的火灾检测数据集 推理识别检测无人机视角下的火灾识别
@[toc]


无人机视角下的火灾检测数据集 共12736张。YOLO适用的txt格式。已划分为训练集、验证集、测试集。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ab73a9c6639a4f0f935cd9bd435176e5.png)

类别：Fire
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4a7d633c3a7b449892b552541bd1cb9c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c6756cd01224e52ac277e8a45ac60c1.png)
使用YOLOv8训练无人机视角下的火灾检测数据集，从环境配置开始，一直到模型的训练和验证。以下是详细的步骤指南：

### 1. 环境配置

#### 1.1 安装CUDA驱动
确保你的显卡支持CUDA，并安装相应的CUDA驱动程序。你可以访问NVIDIA官网下载与你的显卡和操作系统匹配的CUDA驱动。

#### 1.2 安装Anaconda
从[Anaconda官网](https://www.anaconda.com/products/distribution)下载并安装Anaconda。这是一个开源的Python发行版，它包含了大量科学计算、数据分析和机器学习相关的库。

#### 1.3 创建Python虚拟环境
打开Anaconda Prompt（或命令行工具），创建一个新的虚拟环境：
```bash
conda create -n fire_detection python=3.9
conda activate fire_detection
```

#### 1.4 安装必要的依赖项
在激活的虚拟环境中，安装所需的Python包：
```bash
pip install opencv-python-headless PyQt5 ultralytics torch torchvision torchaudio
```
注意：如果需要GPU加速，请确保安装了对应的CUDA版本的PyTorch。

### 2. 数据集准备

确保你的数据集已按照以下结构组织：
```
fire_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
并且标注文件是YOLO格式的txt文件。

### 3. 配置YOLOv8训练

#### 3.1 准备数据配置文件
创建一个`data.yaml`文件，定义数据集路径和类别信息：
```yaml
train: ./fire_dataset/images/train
val: ./fire_dataset/images/val
test: ./fire_dataset/images/test

nc: 1 # 类别数量
names: ['Fire'] # 类别名称
```

#### 3.2 开始训练YOLOv8模型
首先，确保你已经安装了Ultralytics YOLOv8：
```bash
pip install ultralytics
```

然后，运行以下命令开始训练模型。你可以根据自己的需求调整参数。
```bash
yolo task=detect mode=train model=yolov8s.yaml data=data.yaml epochs=100 batch=16 imgsz=640
```
- `model=yolov8s.yaml`：指定要使用的YOLOv8模型架构（如`s`代表small）。
- `epochs=100`：设置训练周期数。
- `batch=16`：设置批量大小。
- `imgsz=640`：设置输入图像尺寸。

### 4. 训练过程中的注意事项

- **监控训练进度**：可以使用TensorBoard来监控训练进度和性能。
- **调整超参数**：根据训练效果适当调整学习率、批次大小等超参数。
- **保存最佳模型**：通常会保存验证集上表现最好的模型权重。

### 5. 使用训练好的模型进行预测

训练完成后，你可以加载模型并对新图片或视频进行预测：
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')  # 替换为你的模型路径

# 对一张图片进行预测
results = model('path_to_your_image.jpg')

# 或者对视频进行预测
results = model('path_to_your_video.mp4')
```

### 6. 完整代码示例

这里提供了一个简单的脚本示例，用于展示如何加载训练好的模型并对其进行测试：
```python
from ultralytics import YOLO
import cv2

def main():
    # 加载训练好的模型
    model = YOLO('runs/detect/train/weights/best.pt')  # 替换为你的模型路径
    
    # 对一张图片进行预测
    results = model('path_to_your_image.jpg')
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取预测框坐标
        confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
        classes = result.boxes.cls.cpu().numpy()  # 获取类别ID
        
        # 读取原图
        img = cv2.imread('path_to_your_image.jpg')
        
        # 在原图上绘制预测框
        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f'{int(cls)} {conf:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

以上文字及代码仅供参考学习使用。
