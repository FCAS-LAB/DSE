#include <math.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <vector>

#include "apis_c.h"

using json = nlohmann::json;
// using namespace InterChiplet;

int srcX = 0, srcY = 0;
int input_size = 0;
std::vector<int> hidden_size;//隐藏层大小
int output_size = 0;//输出层大小
std::vector<std::vector<std::vector<double>>> weight, biases;//权重和偏置
std::vector<int> layer_sizes;//层大小   
std::vector<std::vector<std::vector<double>>> zs, activations;//z和激活值
std::vector<std::vector<double>> a1;//a1
std::vector<std::vector<double>> Randn(int line, int column);//随机生成矩阵

/**
 * @brief 创建一个新文件
 * @param fileName 要创建的文件名
 * @details 以写入模式打开文件，如果成功则输出提示信息，最后关闭文件句柄
 */
void mkfile(char* fileName) {
    FILE* file = fopen(fileName, "w");
    if (file != NULL) std::cout << fileName << "创建成功" << std::endl;
    fclose(file);
}

/**
 * @brief 检查特定格式的文件是否存在
 * @return 文件存在返回1，不存在返回0
 * @details 根据提供的坐标参数构造特定格式的文件名，然后检查该文件是否存在
 */
bool checkfile(int srcX, int srcY, int dstX, int dstY) {
    char* fileName = new char[100];
    sprintf(fileName, "./cpuRead%d_%d_%d_%d", srcX, srcY, dstX, dstY);
    FILE* file = fopen(fileName, "r");
    delete[] fileName;
    if (file == NULL)
        return 0;
    else
        return 1;
}

/**
 * @brief 删除指定文件
 * @param fileName 要删除的文件名
 * @details 使用remove函数删除文件，如果成功则输出删除成功的信息
 */
void delfile(char* fileName) {
    if (remove(fileName) == 0) {
        printf("文件 \"%s\" 已成功删除。\n", fileName);
    }
}

/**
 * @brief 初始化BP神经网络的结构和参数
 * @param Input_size 输入层大小
 * @param Hidden_size 隐藏层大小的向量，可以有多个隐藏层
 * @param Output_size 输出层大小
 * @param SrcX 源X坐标，用于设备间通信
 * @param SrcY 源Y坐标，用于设备间通信
 * @details 初始化神经网络的层大小，并使用He初始化方法初始化权重和偏置
 */
void BPNeuralNetwork(int Input_size, std::vector<int> Hidden_size, int Output_size, int SrcX,
                     int SrcY) {
    srcX = SrcX;  // 设置全局源X坐标
    srcY = SrcY;  // 设置全局源Y坐标
    hidden_size = Hidden_size;  // 设置隐藏层大小
    input_size = Input_size;  // 设置输入层大小
    output_size = Output_size;  // 设置输出层大小
    layer_sizes.push_back(input_size);//输入层大小
    for (size_t i = 0; i < hidden_size.size(); i++) {
        layer_sizes.push_back(hidden_size[i]);//隐藏层大小
    }
    layer_sizes.push_back(output_size);//输出层大小
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {  // He初始化
        std::vector<std::vector<double>> rand_arr = Randn(layer_sizes[i + 1], layer_sizes[i]);//调用Randn函数生成标准正态分布（均值0，标准差1）的随机矩阵
        for (size_t k = 0; k < rand_arr.size(); k++) {
            for (size_t j = 0; j < rand_arr[k].size(); j++) {
                rand_arr[k][j] *= sqrt(2.0 / (layer_sizes[i] > 1 ? layer_sizes[i] : 1));//He初始化 使用ReLU激活函数
                std::cout << rand_arr[k][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "##########################################" << std::endl;
        weight.push_back(rand_arr);//权重矩阵维度：[下一层神经元数, 当前层神经元数]
        biases.push_back(Randn(layer_sizes[i + 1], 1));//偏置矩阵维度：[下一层神经元数, 1]
    }
}

/**
 * @brief 将一维数组转换为二维向量
 * @param V 输入/输出的二维向量结构
 * @param weight_i 输入的一维数组
 * @return 转换后的二维向量
 * @details 按行优先顺序将一维数组的元素复制到二维向量中
 */
std::vector<std::vector<double>> doubleToVector(std::vector<std::vector<double>> V,
                                                double* weight_i) {
    for (size_t i = 0; i < V.size(); i++) {
        // std::vector<double> temp;
        for (size_t j = 0; j < V[i].size(); j++) {
            V[i][j] = (weight_i[j + i * V[0].size()]);
        }
        // V.push_back(temp);
    }
    return V;
}

/**
 * @brief 生成服从标准正态分布的随机矩阵
 * @param line 矩阵的行数
 * @param column 矩阵的列数
 * @return 包含随机值的二维向量
 * @details 使用C++11的随机数生成器创建一个均值为0，标准差为1的正态分布随机矩阵
 */
std::vector<std::vector<double>> Randn(int line, int column) {
    std::random_device rd;  // 用于获取随机种子
    std::mt19937 gen(rd());  // Mersenne Twister伪随机数生成器
    std::normal_distribution<> distribution(0.0, 1.0);  // 均值为0，标准差为1的正态分布
    std::vector<std::vector<double>> result;
    for (int i = 0; i < line; i++) {
        std::vector<double> result_i;
        for (int j = 0; j < column; j++) {
            result_i.push_back(distribution(gen));//随机生成矩阵
        }
        result.push_back(result_i);
    }
    return result;
}

/**
 * @brief 将二维向量转换为一维数组
 * @param V 输入的二维向量
 * @param weight_i 输出的一维数组
 * @details 按行优先顺序将二维向量的元素复制到一维数组中
 */
void vectorToDouble(std::vector<std::vector<double>> V, double* weight_i) {
    for (size_t i = 0; i < V.size(); i++) {
        for (size_t j = 0; j < V[i].size(); j++) {
            weight_i[j + i * V[0].size()] = V[i][j];
        }
    }
}

/**
 * @brief 将双精度浮点数数组转换为整数数组
 * @param mat1 输入的双精度浮点数数组
 * @param mat2 输出的整数数组
 * @param size 数组大小
 * @details 将每个浮点数乘以10^8后转换为整数，用于精度转换
 */
void DoubleToInt(double* mat1, int64_t* mat2, int size) {
    int64_t time = std::pow(10, 8);
    for (int i = 0; i < size; i++) {
        mat2[i] = mat1[i] * time;
    }
}

/**
 * @brief 将整数数组转换为双精度浮点数数组
 * @param mat1 输出的双精度浮点数数组
 * @param mat2 输入的整数数组
 * @param size 数组大小
 * @details 将每个整数除以10^16后转换为浮点数，用于精度恢复
 */
void IntToDouble(double* mat1, int64_t* mat2, int size) {
    double time = std::pow(10, 16);
    for (int i = 0; i < size; i++) {
        mat1[i] = mat2[i] / time;
    }
}

/**
 * @brief 使用GPU进行矩阵乘法计算
 * @param mat1 第一个输入矩阵（一维数组表示）
 * @param mat2 第二个输入矩阵（一维数组表示）
 * @param fst_Row 第一个矩阵的行数
 * @param fst_Col 第一个矩阵的列数
 * @param sec_Row 第二个矩阵的行数
 * @param sec_Col 第二个矩阵的列数
 * @param Res 输出结果矩阵（引用传递）
 * @param dstX 目标设备X坐标
 * @param dstY 目标设备Y坐标
 * @details 将浮点数矩阵转换为整数，发送到GPU设备进行计算，然后接收结果并转换回浮点数
 */
void GpuMultiply(double* mat1, double* mat2, int fst_Row, int fst_Col, int sec_Row, int sec_Col,
                 std::vector<std::vector<double>>& Res, int dstX, int dstY) {
    int64_t* Mat1 = new int64_t[fst_Row * fst_Col];  // 为第一个矩阵分配整数数组
    int64_t* Mat2 = new int64_t[sec_Row * sec_Col];  // 为第二个矩阵分配整数数组
    int64_t* Mat1_size = new int64_t[2];  // 存储第一个矩阵的尺寸
    int64_t* Mat2_size = new int64_t[2];  // 存储第二个矩阵的尺寸
    Mat1_size[0] = fst_Row;  // 设置第一个矩阵的行数
    Mat1_size[1] = fst_Col;  // 设置第一个矩阵的列数
    Mat2_size[0] = sec_Row;  // 设置第二个矩阵的行数
    Mat2_size[1] = sec_Col;  // 设置第二个矩阵的列数
    DoubleToInt(mat1, Mat1, fst_Row * fst_Col);  // 将第一个矩阵转换为整数
    DoubleToInt(mat2, Mat2, sec_Row * sec_Col);  // 将第二个矩阵转换为整数
    std::cout << "hello" << std::endl;
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, Mat1_size, 2 * sizeof(int64_t));  // 发送第一个矩阵的尺寸
    std::cout << "hell0 2" << std::endl;
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, Mat2_size, 2 * sizeof(int64_t));  // 发送第二个矩阵的尺寸
    std::cout << "##########################################" << std::endl;
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, Mat1, fst_Col * fst_Row * sizeof(int64_t));  // 发送第一个矩阵数据
    std::cout << "##########################################" << std::endl;
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, Mat2, sec_Row * sec_Col * sizeof(int64_t));  // 发送第二个矩阵数据
    std::cout << "##########################################" << std::endl;
    bool file = 1;
    while (file == 0) {  // 等待计算完成
        file = checkfile(dstX, dstY, srcX, srcY);
    }

    double* result = new double[fst_Row * sec_Col];  // 为结果分配浮点数数组
    int64_t* Result_2 = new int64_t[fst_Row * sec_Col];  // 为接收的整数结果分配内存
    InterChiplet::receiveMessage(srcX, srcY, dstX, dstY, Result_2,
                                 fst_Row * sec_Col * sizeof(int64_t));  // 接收计算结果
    IntToDouble(result, Result_2, fst_Row * sec_Col);  // 将整数结果转换为浮点数
    // std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::vector<double>> res(fst_Row, std::vector<double>(sec_Col));  // 创建结果矩阵
    Res = doubleToVector(res, result);  // 将一维数组转换为二维向量
    // delete[] fileName;
    // fileName=NULL;
    delete[] result;  // 释放内存
    result = NULL;
    delete[] Result_2;  // 释放内存
    Result_2 = NULL;
    delete[] Mat1;  // 释放内存
    delete[] Mat2;  // 释放内存
    delete[] Mat1_size;  // 释放内存
    delete[] Mat2_size;  // 释放内存
    delete[] mat1;  // 释放内存
    delete[] mat2;  // 释放内存
    Mat1 = NULL;  // 避免悬空指针
    Mat2 = NULL;  // 避免悬空指针
    Mat1_size = NULL;  // 避免悬空指针
    Mat2_size = NULL;  // 避免悬空指针
}

/**
 * @brief GPU矩阵乘法的分块计算实现
 * @param mat1 第一个输入矩阵（一维数组表示）
 * @param mat2 第二个输入矩阵（一维数组表示）
 * @param fst_Row 第一个矩阵的行数
 * @param fst_Col 第一个矩阵的列数
 * @param sec_Row 第二个矩阵的行数
 * @param sec_Col 第二个矩阵的列数
 * @param Res 输出结果矩阵（引用传递）
 * @param gpu_num GPU编号，用于选择计算设备
 * @details 实现了矩阵乘法的分块并行计算，将大矩阵分成多个小块在不同GPU上并行计算
 */
void ToGPU(double* mat1, double* mat2, int fst_Row, int fst_Col, int sec_Row, int sec_Col,
           std::vector<std::vector<double>>& Res, int gpu_num) {
    // 1. 声明存储分块矩阵的数据结构
    std::vector<std::vector<std::vector<double>>> dev1, dev2;  // 存储分块后的两个输入矩阵
    std::vector<std::vector<double>> dev1_, dev2_;  // 临时存储单个分块
    int Col_per_GPU = fst_Col / 3;  // 每个GPU处理的列数（将矩阵平均分成3份）

    // 2. 对第一个矩阵进行分块
    for (int start = 0; start < fst_Col; start += Col_per_GPU) {
        for (int i = 0; i < fst_Row; i++) {
            std::vector<double> dev_temp;  // 存储当前行的一个分块
            // 提取当前分块的元素
            for (int j = start; j < fst_Col && j < start + Col_per_GPU; j++) {
                dev_temp.push_back(mat1[i * fst_Col + j]);
            }
            dev1_.push_back(dev_temp);
        }
        dev1.push_back(dev1_);  // 将当前分块添加到dev1中
        dev1_.clear();  // 清空临时存储，准备下一个分块
    }

    // 3. 对第二个矩阵进行分块
    for (int i = 0; i < sec_Row; i++) {
        std::vector<double> dev_temp;
        for (int j = 0; j < sec_Col; j++) {
            dev_temp.push_back(mat2[i * sec_Col + j]);
        }
        dev2_.push_back(dev_temp);
        // 当达到每个分块的大小或到达矩阵末尾时，保存当前分块
        if ((i + 1) % Col_per_GPU == 0 || i == sec_Row - 1) {
            dev2.push_back(dev2_);
            dev2_.clear();
        }
    }

    // 4. 设置目标GPU设备
    int dstX = 0;
    if (gpu_num == 2) {
        dstX = 1;  // 如果是GPU 2，则使用不同的设备坐标
    }

    // 5. 准备并行计算
    std::vector<std::thread> THREAD;  // 存储计算线程
    std::vector<std::vector<std::vector<double>>> res;  // 存储各个分块的计算结果
    
    // 初始化结果矩阵空间
    for (size_t i = 0; i < dev1.size(); i++) {
        std::vector<std::vector<double>> res_temp(dev1[i].size(),
                                                 std::vector<double>(dev2[i][0].size()));
        res.push_back(res_temp);
    }

    // 6. 创建并行计算线程
    for (size_t i = 0; i < dev1.size(); i++) {
        // 将分块矩阵转换为一维数组格式
        double* Dev1 = new double[dev1[i].size() * dev1[i][0].size()];
        vectorToDouble(dev1[i], Dev1);
        double* Dev2 = new double[dev2[i].size() * dev2[i][0].size()];
        vectorToDouble(dev2[i], Dev2);

        // 创建新线程执行GPU矩阵乘法
        THREAD.push_back(std::thread(GpuMultiply, Dev1, Dev2, dev1[i].size(), dev1[i][0].size(),
                                   dev2[i].size(), dev2[i][0].size(), std::ref(res[i]), dstX,
                                   i + 1));
    }

    // 7. 等待所有线程完成计算
    for (auto& i : THREAD) {
        i.join();
    }

    // 8. 合并计算结果
    Res = res[0];  // 使用第一个分块的结果初始化
    // 将其他分块的结果累加到最终结果中
    for (size_t i = 1; i < res.size(); i++) {
        for (size_t j = 0; j < res[i].size(); j++) {
            for (size_t z = 0; z < res[i][j].size(); z++) {
                Res[j][z] += res[i][j][z];
            }
        }
    }
}

/**
 * @brief 激活函数实现（Leaky ReLU）
 * @param x 输入的二维向量
 * @return 返回经过激活函数处理后的二维向量
 * @details 实现Leaky ReLU激活函数：f(x) = x if x > 0; f(x) = 0.01x if x <= 0
 */
std::vector<std::vector<double>> activate_function(std::vector<std::vector<double>> x) {
    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < x[i].size(); j++) {
            if (x[i][j] <= 0) x[i][j] = 0.01 * x[i][j];  // Leaky ReLU的负数部分
        }
    }
    return x;
}

/**
 * @brief 激活函数的导数（Leaky ReLU导数）
 * @param x 输入的二维向量
 * @return 返回激活函数导数的二维向量
 * @details 计算Leaky ReLU的导数：f'(x) = 1 if x > 0; f'(x) = 0.01 if x <= 0
 */
std::vector<std::vector<double>> activate_function_derivative(std::vector<std::vector<double>> x) {
    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < x[i].size(); j++) {
            if (x[i][j] <= 0)
                x[i][j] = 0.01;  // 负数部分的导数
            else
                x[i][j] = 1;     // 正数部分的导数
        }
    }
    return x;
}

/**
 * @brief 矩阵转置函数，同时更新全局变量a1
 * @param a 输出的一维数组（转置后的结果）
 * @param x 输入的二维向量
 * @param Row 矩阵的行数
 * @param Col 矩阵的列数
 * @details 将二维向量转置并存储为一维数组，同时更新全局变量a1
 */
void T(double* a, std::vector<std::vector<double>> x, int Row, int Col) {
    for (int j = 0; j < Col; j++) {
        for (int i = 0; i < Row; i++) {
            a[i + j * Row] = x[i][j];    // 转置并存储为一维数组
            a1[j][i] = x[i][j];          // 更新全局变量a1
        }
    }
}

/**
 * @brief 矩阵转置函数
 * @param a 输出的一维数组（转置后的结果）
 * @param x 输入的二维向量
 * @param Row 矩阵的行数
 * @param Col 矩阵的列数
 * @details 将二维向量转置并存储为一维数组，不更新全局变量
 */
void T2(double* a, std::vector<std::vector<double>> x, int Row, int Col) {
    for (int j = 0; j < Col; j++) {
        for (int i = 0; i < Row; i++) {
            a[i + j * Row] = x[i][j];    // 转置并存储为一维数组
        }
    }
    // vectorToDouble(x,a);
    // Transpose_GPU(a,x.size(),x[0].size());
}

/**
 * @brief 计算向量的范数(L2范数)
 * @param vec 输入的二维向量
 * @return 返回向量的L2范数
 * @details 计算二维向量所有元素平方和的平方根
 */
double c_norm(const std::vector<std::vector<double>>& vec) {
    double sum_of_squares = 0.0;
    for (const auto& row : vec) {
        for (double x : row) {
            sum_of_squares += x * x;
        }
    }
    return std::sqrt(sum_of_squares);
}

/**
 * @brief 计算二维向量沿指定轴的平均值
 * @param vec 输入的二维向量
 * @param axis 指定的轴(0表示列方向,1表示行方向)
 * @param m 用于计算平均值的除数
 * @return 返回计算结果的二维向量
 * @details 根据指定的轴计算二维向量的平均值,保持结果的维度
 */
std::vector<std::vector<double>> sum_axis(const std::vector<std::vector<double>>& vec, int axis,
                                          int m) {
    std::vector<std::vector<double>> result;

    if (axis == 0) {  // 沿列方向求和
        for (size_t j = 0; j < vec[0].size(); ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < vec.size(); ++i) {
                sum += vec[i][j];
            }
            result.push_back({sum / m});  // 计算平均值
        }
    } else if (axis == 1) {  // 沿行方向求和
        for (const auto& row : vec) {
            double sum = 0.0;
            for (double x : row) {
                sum += x;
            }
            result.push_back({sum / m});  // 计算平均值
        }
    }
    return result;
}

/**
 * @brief 神经网络的前向传播
 * @param x 输入数据
 * @return 返回网络的输出结果
 * @details 实现神经网络的前向传播算法,包括矩阵乘法、偏置加法和激活函数
 */
std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x) {
    int Row = x.size();
    int Col = x[0].size();
    double* A = new double[Col * Row];  // 为输入数据分配内存
    for (int i = 0; i < Col; i++) {
        std::vector<double> a2;
        for (int j = 0; j < Row; j++) {
            a2.push_back(0);
        }
        a1.push_back(a2);
    }
    T(A, x, Row, Col);  // 转置输入数据
    activations.push_back(a1);  // 存储激活值

    // 对每一层进行计算
    for (size_t i = 0; i < weight.size(); i++) {
        double* Weight = new double[weight[i].size() * weight[i][0].size()];
        vectorToDouble(weight[i], Weight);  // 将权重转换为一维数组
        std::vector<std::vector<double>> DotRns;
        ToGPU(Weight, A, weight[i].size(), weight[i][0].size(), Col, Row, DotRns, 1);  // GPU矩阵乘法

        // 添加偏置
        for (size_t m = 0; m < DotRns.size(); m++) {
            for (size_t n = 0; n < DotRns[m].size(); n++) {
                DotRns[m][n] += biases[i][m][0];
            }
        }
        std::vector<std::vector<double>> z = DotRns;
        a1 = activate_function(z);  // 应用激活函数
        Col = a1.size();
        Row = a1[0].size();
        delete[] A;
        A = new double[Row * Col];
        vectorToDouble(a1, A);
        zs.push_back(z);  // 存储中间结果
        activations.push_back(a1);  // 存储激活值
        delete[] Weight;
        Weight = NULL;
    }
    return a1;  // 返回最终输出
}

/**
 * @brief 神经网络的反向传播算法
 * @param x 输入数据
 * @param y 目标输出数据
 * @param learning_rate 学习率
 * @details 实现神经网络的反向传播算法，计算梯度并更新权重和偏置
 */
void backward(std::vector<std::vector<double>> x, std::vector<std::vector<double>>& y,
              double learning_rate) {
    int m = x.size();  // 获取样本数量
    std::vector<std::vector<double>> y_hat = activations[activations.size() - 1];  // 获取网络输出
    double* a = new double[y.size() * y[0].size()];  // 为目标值分配内存
    int Row = y.size();
    int Col = y[0].size();
    T2(a, y, Row, Col);  // 转置目标值矩阵

    // 计算输出层的误差
    std::vector<std::vector<std::vector<double>>> deltas;  // 存储每一层的误差项
    std::vector<std::vector<double>> d_temp;
    for (size_t i = 0; i < y_hat.size(); i++) {
        std::vector<double> temp;
        for (size_t j = 0; j < y_hat[i].size(); j++) {
            temp.push_back(y_hat[i][j] - a[i * y_hat[i].size() + j]);  // 计算预测值与真实值的差
        }
        d_temp.push_back(temp);
    }
    deltas.push_back(d_temp);

    std::vector<std::vector<std::vector<double>>> grads_weights, grads_biases;  // 存储权重和偏置的梯度
    
    // 从后向前计算每一层的梯度
    for (int i = weight.size() - 1; i >= 0; i--) {
        // 计算激活函数的导数
        std::vector<std::vector<double>> act_F = activate_function_derivative(zs[i]);
        std::vector<std::vector<double>> dz(act_F.size(), std::vector<double>(act_F[0].size()));
        
        // 计算误差项
        for (size_t m = 0; m < act_F.size(); m++) {
            for (size_t n = 0; n < act_F[m].size(); n++) {
                dz[m][n] = act_F[m][n] * deltas[deltas.size() - 1][m][n];
            }
        }

        // 梯度裁剪，防止梯度爆炸
        int max_grad_norm = 10;  // 设置梯度的最大范数
        double norm = c_norm(dz);
        if (norm > max_grad_norm) {
            for (size_t i = 0; i < dz.size(); i++) {
                for (size_t j = 0; j < dz[i].size(); j++) {
                    dz[i][j] *= max_grad_norm / norm;  // 按比例缩放梯度
                }
            }
        }

        // 准备进行GPU计算的数据
        double* Dz = new double[dz.size() * dz[0].size()];
        vectorToDouble(dz, Dz);
        double* Activations_i = new double[activations[i].size() * activations[i][0].size()];
        T2(Activations_i, activations[i], activations[i].size(), activations[i][0].size());

        double* Weight = new double[weight[i].size() * weight[i][0].size()];
        T2(Weight, weight[i], weight[i].size(), weight[i][0].size());
        
        // 使用GPU并行计算梯度
        std::vector<std::vector<double>> deltas_pre;  // 前一层的误差
        std::vector<std::vector<double>> dw;  // 权重的梯度
        std::thread t1(ToGPU, Weight, Dz, weight[i][0].size(), weight[i].size(), dz.size(),
                       dz[0].size(), std::ref(deltas_pre), 1);
        std::thread t2(ToGPU, Dz, Activations_i, dz.size(), dz[0].size(), activations[i][0].size(),
                       activations[i].size(), std::ref(dw), 2);
        
        t1.join();
        t2.join();
        
        deltas.push_back(deltas_pre);
        
        // 计算平均梯度
        for (size_t i = 0; i < dw.size(); i++) {
            for (size_t j = 0; j < dw[i].size(); j++) {
                dw[i][j] *= 1.0 / m;  // 除以样本数量得到平均梯度
            }
        }

        // 释放内存
        delete[] Activations_i;
        Activations_i = NULL;
        
        // 计算偏置的梯度
        std::vector<std::vector<double>> db = sum_axis(dz, 1, m);
        grads_weights.push_back(dw);
        grads_biases.push_back(db);
        
        // 释放内存
        delete[] Weight;
        delete[] Dz;
        Weight = NULL;
        Dz = NULL;
    }
    
    // 反转梯度数组，使其与网络层次对应
    std::reverse(grads_weights.begin(), grads_weights.end());
    std::reverse(grads_biases.begin(), grads_biases.end());

    // 更新权重和偏置
    for (size_t i = 0; i < weight.size(); i++) {
        for (size_t j = 0; j < weight[i].size(); j++) {
            for (size_t k = 0; k < weight[i][j].size(); k++) {
                weight[i][j][k] -= learning_rate * grads_weights[i][j][k];  // 权重更新
            }
        }
    }
    for (size_t i = 0; i < biases.size(); i++) {
        for (size_t j = 0; j < biases[i].size(); j++) {
            for (size_t k = 0; k < biases[i][j].size(); k++) {
                biases[i][j][k] -= learning_rate * grads_biases[i][j][k];  // 偏置更新
            }
        }
    }
}

/**
 * @brief 神经网络训练函数
 * @param x 输入训练数据
 * @param y 目标输出数据
 * @param num_iterations 训练迭代次数，默认为1000
 * @param learning_rate 学习率，默认为0.1
 * @details 通过多次迭代进行前向传播和反向传播来训练神经网络
 */
void train(std::vector<std::vector<double>>& x, std::vector<std::vector<double>> y,
           int num_iterations = 1000, double learning_rate = 0.1) {
    for (int i = 0; i < num_iterations; i++) {
        forward(x);  // 前向传播
        backward(x, y, learning_rate);  // 反向传播更新参数
    }
}

/**
 * @brief 神经网络分类预测函数
 * @param x 输入测试数据
 * @param y_size 输出类别数量
 * @param y 真实标签
 * @details 对输入数据进行分类预测，并输出预测结果和真实标签
 */
void predict_classify(std::vector<std::vector<double>> x, int y_size, std::vector<double> y) {
    // 获取网络输出
    std::vector<std::vector<double>> y_hat;
    y_hat = forward(x);  // 前向传播得到预测结果
    
    // 转换数据格式
    double* Y_hat = new double[y_hat.size() * y_hat[0].size()];
    T(Y_hat, y_hat, y_hat.size(), y_hat[0].size());
    y_hat = doubleToVector(y_hat, Y_hat);
    delete[] Y_hat;
    Y_hat = NULL;
    
    std::vector<int> predictions;  // 存储预测的类别
    
    if (y_size > 2) {  // 多分类情况
        for (size_t i = 0; i < y_hat.size(); i++) {
            double Max = y_hat[i][0];
            int max_index = 1;
            // 找出最大概率对应的类别
            for (size_t j = 0; j < y_hat[i].size(); j++) {
                if (Max > y_hat[i][j]) {
                    Max = y_hat[i][j];
                    max_index = j + 1;
                }
            }
            predictions.push_back(max_index);
        }
    } else {  // 二分类情况
        for (size_t i = 0; i < y_hat.size(); i++) {
            int max_index = 0;
            // 使用0.5作为阈值进行二分类
            for (size_t j = 0; j < y_hat[i].size(); j++) {
                if (y_hat[i][j] > 0.5) {
                    max_index = 1;
                }
            }
            predictions.push_back(max_index);
        }
    }
    
    // 输出预测结果
    for (auto i : predictions) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // 输出真实标签
    for (auto i : y) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief 从JSON文件读取数据
 * @param filename JSON文件名
 * @param x_train 训练数据输入
 * @param x_test 测试数据输入
 * @param y_train 训练数据标签
 * @param y_test 测试数据标签
 * @return 读取成功返回true，失败返回false
 * @details 从JSON文件中读取训练和测试数据，并进行one-hot编码转换
 */
bool readDataFromJSON(const std::string& filename, std::vector<std::vector<double>>& x_train,
                      std::vector<std::vector<double>>& x_test,
                      std::vector<std::vector<double>>& y_train,
                      std::vector<std::vector<double>>& y_test) {
    // 打开JSON文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening JSON file." << std::endl;
        return false;
    }

    json data;
    file >> data;

    // 读取训练数据
    for (const auto& row : data["x_train"]) {
        std::vector<double> row_vec;
        for (const auto& val : row) {
            row_vec.push_back(val);
            std::cout << val << " ";
        }
        std::cout << std::endl;
        x_train.push_back(row_vec);
    }

    // 读取测试数据
    for (const auto& row : data["x_test"]) {
        std::vector<double> row_vec;
        for (const auto& val : row) {
            row_vec.push_back(val);
            std::cout << val << " ";
        }
        std::cout << std::endl;
        x_test.push_back(row_vec);
    }

    // 读取训练标签并进行one-hot编码
    for (const auto& val : data["y_train"]) {
        std::vector<double> temp;
        if (val == 1) {
            temp.push_back(1);
            temp.push_back(0);
            temp.push_back(0);
        } else if (val == 2) {
            temp.push_back(0);
            temp.push_back(1);
            temp.push_back(0);
        } else {
            temp.push_back(0);
            temp.push_back(0);
            temp.push_back(1);
        }
        y_train.push_back(temp);
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 读取测试标签并进行one-hot编码
    for (const auto& val : data["y_test"]) {
        std::vector<double> temp;
        if (val == 1) {
            temp.push_back(1);
            temp.push_back(0);
            temp.push_back(0);
        } else if (val == 2) {
            temp.push_back(0);
            temp.push_back(1);
            temp.push_back(0);
        } else {
            temp.push_back(0);
            temp.push_back(0);
            temp.push_back(1);
        }
        y_test.push_back(temp);
        std::cout << val << " ";
    }
    return true;
}

/**
 * @brief 主函数
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序退出状态
 * @details 程序的入口点，初始化神经网络并进行训练
 */
int main(int argc, char** argv) {
    // 创建文件名
    char* fileName = new char[100];
    sprintf(fileName, "start running");
    
    // 设置源设备坐标
    int srcX = 0;
    int srcY = 0;
    
    // 设置神经网络隐藏层结构
    std::vector<int> hidden_size;
    hidden_size.push_back(10);  // 第一个隐藏层10个神经元
    hidden_size.push_back(15);  // 第二个隐藏层15个神经元
    
    // 声明数据存储变量
    std::vector<std::vector<double>> x_train, x_test;
    std::vector<std::vector<double>> y_train, y_test;
    
    // 从JSON文件读取数据
    if (!readDataFromJSON("../temp_data.json", x_train, x_test, y_train, y_test)) {
        std::cout << "数据读取错误" << std::endl;
    }
    
    // 初始化神经网络（输入层13个特征，指定隐藏层，输出层3个类别）
    BPNeuralNetwork(13, hidden_size, 3, srcX, srcY);
    
    // 训练神经网络（迭代1次）
    train(x_train, y_train, 1);
    
    // 清理资源
    delfile(fileName);
    delete[] fileName;
    fileName = NULL;
    
    return 0;
}
