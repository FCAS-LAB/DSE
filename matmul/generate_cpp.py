import math
import os
import subprocess
from datetime import datetime
from ruamel.yaml import YAML, CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

class ChipletMapper:
    """芯片坐标生成器（自动跳过(0,0)保留给sniper）"""
    def __init__(self, num_chips):
        if num_chips <= 0:
            raise ValueError("芯片数量必须大于0")
        
        # 计算网格尺寸（包含保留的(0,0)）
        self.grid_size = math.ceil(math.sqrt(num_chips + 1))  # +1保留(0,0)
        self.coords = []
        
        # 生成坐标网格（跳过0,0）
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) == (0, 0):
                    continue
                if len(self.coords) < num_chips:
                    self.coords.append((x, y))
                else:
                    return

def generate_cpp_code(num_chips):
    """生成完整的矩阵乘法CPP代码"""
    try:
        mapper = ChipletMapper(num_chips)
    except ValueError as e:
        raise e
    
    # 生成发送代码（完整矩阵）
    send_code = []
    for x, y in mapper.coords:
        send_block = f"""
        // 发送到芯片({x}, {y})
        InterChiplet::sendMessage({x}, {y}, idX, idY, 
            A, Row * Col * sizeof(int64_t));
        InterChiplet::sendMessage({x}, {y}, idX, idY,
            B, Row * Col * sizeof(int64_t));"""
        send_code.append(send_block)
    
    # 生成接收代码
    receive_code = []
    c_buffers = []
    for x, y in mapper.coords:
        receive_block = f"""
        // 从芯片({x}, {y})接收结果
        int64_t *C_{x}_{y} = (int64_t *)malloc(Col * sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, {x}, {y},
            C_{x}_{y}, Col * sizeof(int64_t));"""
        receive_code.append(receive_block)
        c_buffers.append(f"C_{x}_{y}")

    # 聚合逻辑
    aggregate_code = "\n        ".join([f"C[i] += {buffer}[i];" for buffer in c_buffers])

    # CPP代码模板
    cpp_code = f"""/*
 * @Author: auto-generated
 * @Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * @LastEditors: auto-generated
 * @LastEditTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * @Description: 自动生成的矩阵乘法内核（芯片数={num_chips}）
 */
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "apis_c.h"

#define Row 100
#define Col 100

int idX, idY;

int main(int argc, char **argv) {{
    // 参数校验
    if (argc < 3) {{
        std::cerr << "Usage: " << argv[0] << " <idX> <idY>" << std::endl;
        return 1;
    }}
    
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    // 内存分配（完整矩阵）
    int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *C = (int64_t *)malloc(sizeof(int64_t) * Col);
    if (!A || !B || !C) {{
        std::cerr << "内存分配失败" << std::endl;
        return 2;
    }}

    // 矩阵初始化
    for (int i = 0; i < Row * Col; ++i) {{
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }}

    // 数据分发（发送到{num_chips}个芯片）
    {''.join(send_code)}

    // 接收结果
    {''.join(receive_code)}

    // 聚合结果
    for (int i = 0; i < Col; ++i) {{
        C[i] = 0;
        {aggregate_code}
    }}

    // 释放内存
    free(A);
    free(B);
    free(C);
    {''.join([f'free({buffer});' for buffer in c_buffers])}

    return 0;
}}
"""
    return cpp_code

def generate_yml_config(num_chips):
    """严格匹配目标格式的YAML生成函数"""
    mapper = ChipletMapper(num_chips)
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)  # 精确控制缩进

    config = CommentedMap()
    
    # ========== Phase1配置 ==========
    phase1 = CommentedSeq()
    
    # 添加gpgpusim进程（带Process注释）
    for i, (x, y) in enumerate(mapper.coords):
        proc = CommentedMap()
        proc.yaml_set_comment_before_after_key(
            "cmd", 
            before=f"\n  # Process {i}"
        )
        proc["cmd"] = dq("$BENCHMARK_ROOT/bin/matmul_cu")
        proc["args"] = CommentedSeq([dq(str(x)), dq(str(y))])
        proc["args"].fa.set_flow_style()  # 保持紧凑数组格式
        proc["log"] = dq(f"gpgpusim.{x}.{y}.log")
        proc["is_to_stdout"] = False
        proc["clock_rate"] = 1
        proc["pre_copy"] = dq("$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*")
        phase1.append(proc)
    
    # 添加sniper进程（最后一个Process）
    sniper_proc = CommentedMap()
    sniper_proc.yaml_set_comment_before_after_key(
        "cmd", 
        before=f"\n  # Process {len(mapper.coords)}"
    )
    sniper_proc["cmd"] = dq("$SIMULATOR_ROOT/snipersim/run-sniper")
    sniper_proc["args"] = CommentedSeq([
        dq("--"), 
        dq("$BENCHMARK_ROOT/bin/matmul_c"),
        dq("0"), 
        dq("0")
    ])
    sniper_proc["args"].fa.set_flow_style()
    sniper_proc["log"] = dq("sniper.0.0.log")
    sniper_proc["is_to_stdout"] = False
    sniper_proc["clock_rate"] = 1
    phase1.append(sniper_proc)

    # ========== Phase2配置 ==========
    phase2 = CommentedSeq()
    popnet_proc = CommentedMap()
    popnet_proc.yaml_set_comment_before_after_key(
        "cmd", 
        before="\n  # Process 0"
    )
    popnet_proc["cmd"] = dq("$SIMULATOR_ROOT/popnet_chiplet/build/popnet")
    args = [
        "-A", "2", "-c", "2", "-V", "3", "-B", "12", "-O", "12",
        "-F", "4", "-L", "1000", "-T", "10000000", "-r", "1",
        "-I", "../bench.txt", "-R", "0", "-D", "../delayInfo.txt", "-P"
    ]
    popnet_proc["args"] = CommentedSeq([dq(arg) for arg in args])
    popnet_proc["args"].fa.set_flow_style()  # 保持紧凑格式
    popnet_proc["log"] = dq("popnet_0.log")
    popnet_proc["is_to_stdout"] = False
    popnet_proc["clock_rate"] = 1
    phase2.append(popnet_proc)

    # ========== 构建完整配置 ==========
    config["phase1"] = phase1
    config["phase2"] = phase2
    config["bench_file"] = dq("./bench.txt")
    config["delayinfo_file"] = dq("./delayInfo.txt")

    # ========== 添加全局注释 ==========
    config.yaml_set_comment_before_after_key("phase1", 
        before="# Phase 1 configuration.",
        indent=0)
    
    config.yaml_set_comment_before_after_key("phase2", 
        before="\n# Phase 2 configuration.",
        indent=0)
    
    config.yaml_set_comment_before_after_key("bench_file", 
        before="\n# File configuration. (Not used yet)",
        indent=0)

    return config

def execute_chiplet(yml_file):
    """
    使用 chiplet 执行 YAML 文件。

    参数:
    yml_file (str): 要执行的 YAML 文件路径。
    """
    # 构建执行 chiplet 的命令
    cmd = [
        "../../interchiplet/bin/interchiplet",#TODO 任务路径
        "--cwd",
        os.path.dirname(yml_file),  # 使用 YAML 文件所在的目录作为工作目录
        yml_file
    ]
    # 执行命令
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    try:
        num_chips = int(input("请输入芯片数量（仅gpgpusim进程）: "))
        if num_chips <= 0:
            raise ValueError("芯片数量必须大于0")
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 生成CPP文件
        cpp_path = os.path.join("output", f"matmul_{num_chips}.cpp")
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write(generate_cpp_code(num_chips))
        
        # 生成YAML文件
        yaml = YAML()
        yaml_path = os.path.join("output", f"matmul_{num_chips}.yml")
        with open(yaml_path, 'w') as f:
            yaml.dump(generate_yml_config(num_chips), f)
        
        print(f"生成成功！文件已保存至：\n  {cpp_path}\n  {yaml_path}")
        
    except ValueError as e:
        print(f"输入错误: {str(e)}")
    except Exception as e:
        print(f"运行时错误: {str(e)}")