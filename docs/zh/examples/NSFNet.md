# NSFNet 

<a href="https://aistudio.baidu.com/studio/project/partial/verify/6832363/da4e1b9b08f14bd4baf9b8b6922b5b7e" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py

    # VP_NSFNet2
    # linux
    wget 
    # windows
    # curl 
    python VP_NSFNet2.py

    # VP_NSFNet3
    python VP_NSFNet2.py
    ```

=== "模型评估命令"

## 1. 背景简介

## 2. 问题定义

## 3. 问题求解
为节约篇幅，问题求解以NSFNet3为例
### 3.1 模型构建

### 3.2 数据生成

### 3.3 约束构建

### 3.4 评估器构建

### 3.5 超参数设定

### 3.6 优化器构建

### 3.7 自定义 loss

### 3.8 模型训练与评估

## 4. 完整代码
``` py linenums="1" title="epnn.py"
--8<--
examples/NSFNet/VP_NSFNet1.py
examples/NSFNet/VP_NSFNet2.py
examples/NSFNet/VP_NSFNet3.py
--8<--
```
## 5. 结果展示
### NSFNet1:
| size 4*50 | paper  | code(without BFGS) | PaddleScience  |
|-------------------|--------|--------------------|---------|
| u                 | 0.084% | 0.062%             | 0.055%  |
| v                 | 0.425% | 0.431%             | 0.399%  |
### NSFNet2:
| size 10*100 t=0| paper  | code(without BFGS) | PaddleScience  |
|-------------------|--------|--------------------|---------|
| u                 | /| 0.403%         | 0.138%  |
| v                 | / | 1.5%             |  0.488% |

![image](https://github.com/DUCH714/hackthon5th53/blob/develop/examples/NSFNet/fig/movie.gif)

### NSFNet3:
| size 10*100 t=1 | paper  | code(without BFGS) | PaddleScience  |
|-------------------|--------|--------------------|---------|
| u                 | 0.426%| /         | /  |
| v                 | 0.366% | /            | /  |
| w                 | 0.587% | /            | /  |

## 6. 参考资料
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
