# PaddleOCRv5_ncnn
PaddleOCRv5 在Visual Studio中进行图片OCR检测（ncnn框架+open-mobile实现)，尝试对nihui的[ncnn-android-ppocrv5](https://github.com/nihui/ncnn-android-ppocrv5)检测算法的剥离与移植。



### 写在前面

本仓库代码是基于nihui的[ncnn-android-ppocrv5](https://github.com/nihui/ncnn-android-ppocrv5)项目代码而修改的，原仓库代码是部署在Android端的，对于想在其他环境部署来说，需要进行代码剥离和移植。本仓库的代码即是执行该次尝试，尝试在Windows短的Visual Studio中部署该算法。



### 环境配置

本仓库代码的运行环境如下：

- Visual Studio 2019
- [ncnn-20250503-windows-vs2019](https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-windows-vs2019.zip)

- [opencv-mobile-3.4.20-windows-vs2019](https://github.com/nihui/opencv-mobile/releases/download/v33/opencv-mobile-3.4.20-windows-vs2019.zip)

**注：并不需要和本仓库代码的配置环境保持一致，可以根据Visual Studio的版本，下载对应的ncnn和opencv-mobile的版本即可。**



### 推理设置

（1）先在Visual Studio新建一个空白工程，将本仓库代码放到该工程中。

![01](C:\Users\HIT-HAYES\Desktop\docs\01.png)

（2）在工程中载入推理需要依赖的库。

![04](C:\Users\HIT-HAYES\Desktop\docs\02.png)![03](C:\Users\HIT-HAYES\Desktop\docs\03.png)![05](C:\Users\HIT-HAYES\Desktop\docs\04.png)

需要添加的依赖项如下：

```shell
ncnn.lib
GenericCodeGen.lib
glslang.lib
glslang-default-resource-limits.lib
MachineIndependent.lib
OSDependent.lib
SPIRV.lib
opencv_core3420.lib
opencv_features2d3420.lib
opencv_highgui3420.lib
opencv_imgproc3420.lib
opencv_photo3420.lib
opencv_video3420.lib
```

![02](C:\Users\HIT-HAYES\Desktop\docs\05.png)

（3）选择开始执行，应该在工程的**Release**的目录下可以成功地生成`.exe`文件。将`weights`文件夹和测试图像复制到Release下的路径。

![06](C:\Users\HIT-HAYES\Desktop\docs\06.png)

（4）执行推理。

![06](C:\Users\HIT-HAYES\Desktop\docs\07.png)

推理图片的指令如下：

```sh
PaddleOCRv5.exe single japan.png     // 推理图像
PaddleOCRv5.exe folder images        // 推理文件夹(多张图像)  
```

输出结果会保存在`output`文件夹下。



### 推理结果

推理结果如下：

![motto](C:\Users\HIT-HAYES\Desktop\docs\motto.png)

![test](C:\Users\HIT-HAYES\Desktop\docs\test.png)

![result](C:\Users\HIT-HAYES\Desktop\docs\result.jpg)



### 写在后面

- 由于原本的opencv不支持中文和其他语言显示，因此使用的是nihui发布的open-mobile，该库可以支持简单的opencv操作，同时支持中文日文等的显示。但是不知道为什么，我无法在windows端成功地调用电脑的摄像头，也就没有办法进行实时推理的测试。
- 有尝试在ubuntu平台移植算法，由于opencv-mobile库的原因，无法成功编译库，因此也没有办法完整正常推理。
- 本仓库代码对nihui原始的仓库做了部分修改，选择通过读取`.txt`字符文件的方式读取字符，而源代码中是将字符一整个写在`.h`文件中，我觉得会增加编译负担，就进行了修改。

创作不易，如果觉得这个仓库还可以的话，麻烦给一个star，这就是对我最大的鼓励。



### Reference

- [ncnn-android-ppocrv5](https://github.com/nihui/ncnn-android-ppocrv5)
- [QT-YOLO-OCR-CPP](https://github.com/WYQ-Github/QT-YOLO-OCR-CPP)

