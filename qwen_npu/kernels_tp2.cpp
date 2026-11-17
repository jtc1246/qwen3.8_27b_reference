// 双芯张量并行版 kernel: 与单芯同一份源码, head 维度减半 (ascendc 编译框架不读 target 宏定义, 用包装文件固定宏)
#define QWEN_TP 2
#include "kernels_npu.cpp"
