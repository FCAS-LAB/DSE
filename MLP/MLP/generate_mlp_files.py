import math
import os
import subprocess
from datetime import datetime
from ruamel.yaml import YAML, CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq
import re # 用于替换函数体

# --- Chiplet 坐标生成器 (复用 generate_cpp.py 的逻辑) ---
class ChipletMapper:
    """芯片坐标生成器（自动跳过(0,0)保留给sniper/主CPU）"""
    def __init__(self, num_chips):
        if num_chips <= 0:
            raise ValueError("GPU芯片数量必须大于0")

        # 计算网格尺寸（包含保留的(0,0)）
        # +1 for the (0,0) reserved for the main CPU process
        self.grid_size = math.ceil(math.sqrt(num_chips + 1))
        self.coords = []

        # 生成坐标网格（跳过0,0）
        count = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) == (0, 0):
                    continue
                if count < num_chips:
                    self.coords.append((x, y))
                    count += 1
                else:
                    # Stop generating coordinates once we have enough
                    return
        # If the loop finishes and we don't have enough chips, something is wrong
        if count < num_chips:
             raise RuntimeError(f"无法为 {num_chips} 个芯片生成足够的唯一坐标（网格大小 {self.grid_size}x{self.grid_size}）")


# --- C++ 代码生成逻辑 ---

def generate_togu_function_body(num_gpu_chiplets, mapper, split_mode='vertical'):
    """根据GPU芯粒数量和切分模式生成ToGPU函数的主体代码"""
    if num_gpu_chiplets <= 0:
        return "// 无GPU芯片，ToGPU函数保持为空或执行CPU计算（未实现）\\n"

    # --- 通用部分 ---
    common_setup = f"""
    std::vector<std::thread> THREAD;              // 存储计算线程
    // 注意：结果的结构取决于切分模式
    std::vector<std::vector<std::vector<double>>> res(num_gpu_chiplets); // 预分配空间，假设最多 num_gpu_chiplets 个有效分块
    int actual_num_splits = 0; // 记录实际创建的线程/分块数
    """

    # --- 模式特定的代码生成 ---
    if split_mode == 'vertical':
        split_logic = f"""
    // --- 垂直切分 (mat1按列, mat2按行) ---
    std::vector<std::vector<std::vector<double>>> dev1, dev2; // 存储分块后的两个输入矩阵
    std::vector<std::vector<double>> dev1_, dev2_;          // 临时存储单个分块

    int num_splits = {num_gpu_chiplets};
    if (num_splits <= 0) num_splits = 1;

    // 基于 mat1 的列数进行切分
    int Col_per_GPU = (fst_Col + num_splits - 1) / num_splits;
    if (Col_per_GPU == 0 && fst_Col > 0) Col_per_GPU = 1;
    if (fst_Col == 0) Col_per_GPU = 0;

    int current_col_start = 0;
    // 对 mat1 按列分块
    for (int split_idx = 0; split_idx < num_splits && current_col_start < fst_Col; ++split_idx) {{
        dev1_.clear();
        int current_col_end = std::min(current_col_start + Col_per_GPU, fst_Col);
        int current_block_cols = current_col_end - current_col_start;
        if (current_block_cols <= 0) continue;

        for (int i = 0; i < fst_Row; ++i) {{
            std::vector<double> dev_temp;
            dev_temp.reserve(current_block_cols);
            for (int j = current_col_start; j < current_col_end; ++j) {{
                dev_temp.push_back(mat1[i * fst_Col + j]);
            }}
            dev1_.push_back(dev_temp);
        }}
        if (!dev1_.empty()) dev1.push_back(dev1_); // 添加有效分块
        current_col_start = current_col_end;
    }}

    // 对 mat2 按行分块 (行数由 mat1 的列块决定)
    int current_row_start = 0;
    for (int split_idx = 0; split_idx < dev1.size() && current_row_start < sec_Row; ++split_idx) {{ // 使用 dev1 的实际大小
        dev2_.clear();
        // 检查 dev1[split_idx] 是否为空以及是否有列
        if (split_idx >= dev1.size() || dev1[split_idx].empty() || dev1[split_idx][0].empty()) {{
             std::cerr << "Warning: Skipping mat2 split for empty or invalid mat1 block " << split_idx << std::endl;
             // 需要添加一个空的 dev2 块以保持同步，或者后续处理跳过
             dev2.push_back({{}}); // 添加空块
             continue;
        }}
        // 注意：这里 current_block_rows 应该等于 dev1[split_idx][0].size()
        int current_block_rows = dev1[split_idx][0].size();
        int current_row_end = std::min(current_row_start + current_block_rows, sec_Row);
        if (current_row_end - current_row_start <= 0) {{
             dev2.push_back({{}}); // 添加空块
             continue; // 如果行块无效则跳过
        }}

        for (int i = current_row_start; i < current_row_end; ++i) {{
             std::vector<double> dev_temp;
             dev_temp.reserve(sec_Col);
             for(int j = 0; j < sec_Col; ++j) {{
                dev_temp.push_back(mat2[i * sec_Col + j]);
             }}
            dev2_.push_back(dev_temp);
        }}
         if (!dev2_.empty()) dev2.push_back(dev2_); // 添加有效分块
         else dev2.push_back({{}}); // 如果循环未添加任何内容，则添加空块
        current_row_start = current_row_end;
    }}
    // 如果 dev1 比 dev2 多，补齐 dev2
    while(dev2.size() < dev1.size()) {{
        dev2.push_back({{}});
    }}


    // 确保分块数量一致
     if (dev1.size() != dev2.size()) {{
         std::cerr << "Error: Vertical split sizes for mat1 (" << dev1.size()
                   << ") and mat2 (" << dev2.size() << ") do not match after processing!" << std::endl;
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0)); // 返回零矩阵
         return;
     }}
     actual_num_splits = dev1.size(); // 使用实际分块数

    // 初始化结果矩阵空间 (垂直切分模式下，所有结果矩阵大小应一致)
    // 找到第一个有效的分块来确定维度
    int res_rows = fst_Row; // 默认等于 mat1 的行数
    int res_cols = sec_Col; // 默认等于 mat2 的列数
    bool dims_determined = false;
    for(int i=0; i < actual_num_splits; ++i) {{
        if (!dev1[i].empty() && !dev2[i].empty() && !dev2[i][0].empty()) {{
            res_rows = dev1[i].size();
            res_cols = dev2[i][0].size();
            dims_determined = true;
            break;
        }}
    }}

    if (!dims_determined && actual_num_splits > 0) {{
        std::cerr << "Warning: Could not determine result dimensions from splits in vertical mode. Using default." << std::endl;
        // Res 已经在函数开始时初始化
    }}

    if (actual_num_splits > 0) {{
         for (int i = 0; i < actual_num_splits; ++i) {{
              // 为有效分块初始化结果空间，无效分块保持 res[i] 为空
              if (!dev1[i].empty() && !dev2[i].empty() && !dev2[i][0].empty()) {{
                 res[i].assign(res_rows, std::vector<double>(res_cols, 0.0));
              }}
              // else res[i] 保持默认构造（可能为空）
          }}
    }} else {{
         std::cerr << "Warning: No splits generated in vertical mode." << std::endl;
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0));
         return;
    }}
    """

        thread_creation_code = """
    // --- 创建并行计算线程 (垂直切分) ---
    int valid_thread_count = 0;
    for (int i = 0; i < actual_num_splits; ++i) {
        // 检查分块是否有效
        if (dev1[i].empty() || dev1[i][0].empty() || dev2[i].empty() || dev2[i][0].empty()) {
            std::cerr << "Warning: Skipping thread creation for empty or invalid vertical split block " << i << std::endl;
            // res[i] 应该已经为空或未初始化，聚合时会跳过
            continue;
        }

        // 检查维度是否匹配: dev1[i]的列数 == dev2[i]的行数
        if (dev1[i][0].size() != dev2[i].size()) {
            std::cerr << "Error: Dimension mismatch for vertical split block " << i
                      << ". mat1 block columns (" << dev1[i][0].size()
                      << ") != mat2 block rows (" << dev2[i].size() << "). Skipping thread." << std::endl;
            res[i].clear(); // 标记为无效结果
            continue;
        }

        // 检查 mapper.coords 是否足够
        if (valid_thread_count >= mapper.coords.size()) {
             std::cerr << "Error: Not enough coordinates in mapper for valid split " << i << " (needs coordinate index " << valid_thread_count << "). Stopping thread creation." << std::endl;
             break;
        }
        int dstX = mapper.coords[valid_thread_count].first;
        int dstY = mapper.coords[valid_thread_count].second;

        // 将分块矩阵转换为一维数组
        size_t dev1_size = dev1[i].size() * dev1[i][0].size();
        size_t dev2_size = dev2[i].size() * dev2[i][0].size();
        double* Dev1 = new double[dev1_size];
        vectorToDouble(dev1[i], Dev1);
        double* Dev2 = new double[dev2_size];
        vectorToDouble(dev2[i], Dev2);

        // 获取分块维度信息
        int current_fst_Row = dev1[i].size();     // 分块 mat1 的行数 (等于原始 mat1 行数)
        int current_fst_Col = dev1[i][0].size();  // 分块 mat1 的列数
        int current_sec_Row = dev2[i].size();     // 分块 mat2 的行数 (等于 current_fst_Col)
        int current_sec_Col = dev2[i][0].size();  // 分块 mat2 的列数 (等于原始 mat2 列数)

        // 创建线程
        THREAD.emplace_back(GpuMultiply, Dev1, Dev2,
                           current_fst_Row, current_fst_Col,
                           current_sec_Row, current_sec_Col,
                           std::ref(res[i]), // 结果写入 res[i]
                           dstX, dstY);
        valid_thread_count++; // 增加有效线程计数
    }
    actual_num_splits = valid_thread_count; // 更新实际创建的线程数
    """

        join_aggregate_code = f"""
    // --- 等待线程完成 (通用) ---
    for (auto& th : THREAD) {{
        if (th.joinable()) {{
            th.join();
        }}
    }}

    // --- 合并计算结果 (垂直切分 - 累加) ---
    bool first_valid_result = true;
    // 再次检查维度，以防 GpuMultiply 返回了意外的大小
    int expected_rows = -1;
    int expected_cols = -1;

    for (int i = 0; i < res.size(); ++i) {{ // 遍历所有可能的 res 槽位
        if (res[i].empty() || res[i][0].empty()) {{
             continue; // 跳过无效或空结果块
         }}

        if (first_valid_result) {{
             Res = res[i]; // 使用第一个有效结果初始化 Res
             expected_rows = Res.size();
             expected_cols = Res[0].size();
             first_valid_result = false;
        }} else {{
            // 检查维度是否匹配第一个有效结果
            if (res[i].size() != expected_rows || res[i][0].size() != expected_cols) {{
                std::cerr << "Warning: Skipping aggregation for result block " << i
                          << " due to mismatched dimensions (" << res[i].size() << "x" << res[i][0].size()
                          << " vs expected " << expected_rows << "x" << expected_cols << ")." << std::endl;
                continue;
            }}
            // 累加结果
            for (size_t j = 0; j < res[i].size(); ++j) {{
                for (size_t z = 0; z < res[i][j].size(); ++z) {{
                    Res[j][z] += res[i][j][z];
                }}
            }}
        }}
    }}

    if (first_valid_result) {{ // 如果循环结束都没有找到有效结果
         std::cerr << "Warning: No valid results found after threads finished in vertical mode. Result matrix is zero." << std::endl;
         // 确保 Res 至少有正确的外部维度，即使内容为 0
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0));
    }}
    // --- 清理工作 (GpuMultiply 已处理 Dev1, Dev2) ---
    """

    elif split_mode == 'horizontal':
        split_logic = f"""
    // --- 水平切分 (mat1按行, mat2不切分) ---
    std::vector<std::vector<std::vector<double>>> dev1; // 只存储分块后的 mat1
    std::vector<std::vector<double>> dev1_;          // 临时存储单个 mat1 分块

    // 检查 mat2 是否有效 (必须在切分 mat1 之前，因为需要 sec_Row 进行乘法检查)
    if (fst_Col != sec_Row) {{
         std::cerr << "Error: Matrix multiplication dimension mismatch for horizontal split. "
                   << "mat1 columns (" << fst_Col << ") != mat2 rows (" << sec_Row << ")." << std::endl;
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0)); // 返回零矩阵
         return;
    }}
    if (sec_Col == 0) {{
         std::cerr << "Error: mat2 has zero columns in horizontal split mode." << std::endl;
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0)); // 返回零矩阵
         return;
    }}

    int num_splits = {num_gpu_chiplets};
    if (num_splits <= 0) num_splits = 1;

    // 基于 mat1 的行数进行切分
    int Row_per_GPU = (fst_Row + num_splits - 1) / num_splits;
    if (Row_per_GPU == 0 && fst_Row > 0) Row_per_GPU = 1;
    if (fst_Row == 0) {{
        std::cerr << "Warning: mat1 has zero rows in horizontal split mode." << std::endl;
        Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0)); // 结果是 0 行
        return;
    }}

    int current_row_start = 0;
    // 对 mat1 按行分块
    for (int split_idx = 0; split_idx < num_splits && current_row_start < fst_Row; ++split_idx) {{
        dev1_.clear();
        int current_row_end = std::min(current_row_start + Row_per_GPU, fst_Row);
        int current_block_rows = current_row_end - current_row_start;
        if (current_block_rows <= 0) continue;

        for (int i = current_row_start; i < current_row_end; ++i) {{
            std::vector<double> dev_temp;
            dev_temp.reserve(fst_Col); // 行的长度是 mat1 的列数
            for (int j = 0; j < fst_Col; ++j) {{
                 // 注意 mat1 是一维存储的
                dev_temp.push_back(mat1[i * fst_Col + j]);
            }}
            dev1_.push_back(dev_temp);
        }}
         if (!dev1_.empty()) dev1.push_back(dev1_); // 添加有效分块
        current_row_start = current_row_end;
    }}
    actual_num_splits = dev1.size(); // 使用实际分块数


    // 初始化结果矩阵空间 (水平切分模式下，每个结果块行数不同，列数相同)
    bool allocation_needed = false;
    if (actual_num_splits > 0) {{
        for (int i = 0; i < actual_num_splits; ++i) {{
             if (dev1[i].empty() || dev1[i][0].empty()) {{
                  // res[i] 保持默认构造 (空)
                  continue;
             }}
             int res_rows = dev1[i].size(); // 结果行数等于该分块的行数
             int res_cols = sec_Col;       // 结果列数等于 mat2 的列数
             if (res_rows > 0 && res_cols > 0) {{
                 res[i].assign(res_rows, std::vector<double>(res_cols, 0.0));
                 allocation_needed = true;
             }}
             // else res[i] 保持空
         }}
    }} else {{
        std::cerr << "Warning: No mat1 splits generated in horizontal mode." << std::endl;
        Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0));
        return;
    }}
    if (!allocation_needed && actual_num_splits > 0) {{
        std::cerr << "Warning: No valid result space allocated for any split in horizontal mode." << std::endl;
         Res.assign(fst_Row, std::vector<double>(sec_Col, 0.0));
         return; // 没有可计算的分块
    }}
    """
        # mat2 需要转换为 double* 格式一次，传递给所有线程
        # 或者 GpuMultiply 能接受 vector<double>& / const double*?
        # 为保持一致性，我们还是在循环内创建 mat2 的 double* 副本
        # 注意：这会导致 mat2 数据被复制多次，但避免了复杂的生命周期管理
        # 更好的方法是只转换一次，传递 const double*，并修改 GpuMultiply 不 delete 它
        # 但这里我们暂时保持 GpuMultiply 不变，每次都复制

        thread_creation_code = """
    // --- 创建并行计算线程 (水平切分) ---
    // 先将完整的 mat2 转为 double* (在循环外只做一次，避免重复转换)
    size_t full_mat2_size = sec_Row * sec_Col;
    double* Full_Dev2_orig = new double[full_mat2_size];
    // TODO: 确认 mat2 是一维数组还是二维 vector
    // 假设 mat2 是一维数组，如果不是，需要先转换
    // 假设 vectorToDouble 可以处理这种情况，或者需要一个类似的函数
    // for(size_t k=0; k < full_mat2_size; ++k) Full_Dev2_orig[k] = mat2[k];
    // 假设 mat2 是 std::vector<double> flattened_mat2;
    std::copy(mat2.begin(), mat2.end(), Full_Dev2_orig);


    int valid_thread_count = 0;
    for (int i = 0; i < actual_num_splits; ++i) {
        // 检查 mat1 分块是否有效 和 对应的 res 空间是否已分配
        if (dev1[i].empty() || dev1[i][0].empty() || res[i].empty() || res[i][0].empty()) {
            std::cerr << "Warning: Skipping thread creation for empty/invalid horizontal split block " << i << " or unallocated result space." << std::endl;
            continue;
        }

         // 检查 mapper.coords 是否足够
        if (valid_thread_count >= mapper.coords.size()) {
             std::cerr << "Error: Not enough coordinates in mapper for valid split " << i << " (needs coordinate index " << valid_thread_count << "). Stopping thread creation." << std::endl;
             break;
        }
        int dstX = mapper.coords[valid_thread_count].first;
        int dstY = mapper.coords[valid_thread_count].second;

        // 将 mat1 分块转换为一维数组
        size_t dev1_size = dev1[i].size() * dev1[i][0].size();
        double* Dev1 = new double[dev1_size];
        vectorToDouble(dev1[i], Dev1);

        // 为当前线程创建指向完整 mat2 数据的指针 (GpuMultiply会delete)
        // 因此，我们需要复制 Full_Dev2_orig 的数据给 GpuMultiply
        double* Dev2_copy = new double[full_mat2_size];
        std::copy(Full_Dev2_orig, Full_Dev2_orig + full_mat2_size, Dev2_copy);

        // 获取维度信息
        int current_fst_Row = dev1[i].size();     // 分块 mat1 的行数
        int current_fst_Col = dev1[i][0].size();  // 分块 mat1 的列数 (等于原始 mat1 列数)
        int current_sec_Row = sec_Row;            // 完整 mat2 的行数
        int current_sec_Col = sec_Col;            // 完整 mat2 的列数

        // GpuMultiply 需要计算 mat1_slice * mat2
        // 输入维度是 (current_fst_Row, current_fst_Col) * (current_sec_Row, current_sec_Col)
        // 要求 current_fst_Col == current_sec_Row (已在 split_logic 中检查)
        // 输出维度是 (current_fst_Row, current_sec_Col)
        THREAD.emplace_back(GpuMultiply, Dev1, Dev2_copy, // 传递 mat1 块 和 完整 mat2 的副本
                           current_fst_Row, current_fst_Col,
                           current_sec_Row, current_sec_Col,
                           std::ref(res[i]), // 结果是 Res 的一个水平分块
                           dstX, dstY);
        valid_thread_count++; // 增加有效线程计数
    }
    actual_num_splits = valid_thread_count; // 更新实际创建的线程数
    delete[] Full_Dev2_orig; // 清理在循环外创建的 mat2 原始副本
    """

        join_aggregate_code = f"""
    // --- 等待线程完成 (通用) ---
    for (auto& th : THREAD) {{
        if (th.joinable()) {{
            th.join();
        }}
    }}

    // --- 合并计算结果 (水平切分 - 拼接/堆叠) ---
    // 1. 计算最终 Res 的总行数 (等于 mat1 的行数) 和列数 (等于 mat2 的列数)
    int total_rows = fst_Row;
    int total_cols = sec_Col;
    // 只有在有实际计算的情况下才调整 Res 大小
    if (total_rows > 0 && total_cols > 0) {{
        Res.assign(total_rows, std::vector<double>(total_cols, 0.0)); // 初始化最终结果矩阵
    }} else {{
        Res.clear(); // 如果输入维度导致结果为空，则清空
        std::cerr << "Warning: Final result matrix dimensions are zero or invalid in horizontal mode." << std::endl;
        return; // 无需聚合
    }}


    // 2. 将各个分块的结果复制到 Res 的正确位置
    int current_res_row_offset = 0;
    bool valid_result_found = false;
    // 使用 dev1 的分块信息来指导聚合，因为 res[i] 可能因错误而为空
    int split_idx_for_res = 0; // 追踪原始分块索引
    for (const auto& mat1_block : dev1) {{ // 遍历原始 mat1 分块顺序
        if (split_idx_for_res >= res.size()) break; // 防止越界

        // 检查对应的 res[split_idx_for_res] 是否有效
        if (res[split_idx_for_res].empty() || res[split_idx_for_res][0].empty()) {{
             std::cerr << "Warning: Skipping empty/invalid result block corresponding to mat1 split " << split_idx_for_res << " during horizontal aggregation." << std::endl;
             // 仍然需要增加偏移量，因为这个分块在概念上占用了行数
             if (!mat1_block.empty()) {{
                current_res_row_offset += mat1_block.size();
             }}
             split_idx_for_res++;
             continue; // 跳过空/无效结果块
         }}

        // 检查列数是否匹配最终结果的列数
         if (res[split_idx_for_res][0].size() != total_cols) {{
              std::cerr << "Error: Result block for mat1 split " << split_idx_for_res << " has mismatched column count (" << res[split_idx_for_res][0].size()
                        << " vs expected " << total_cols << "). Aggregation might be incorrect." << std::endl;
              // 这里选择继续聚合，但标记错误，或者可以选择停止
              // 增加偏移量并跳过复制?
              if (!mat1_block.empty()) {{
                current_res_row_offset += mat1_block.size();
              }}
              split_idx_for_res++;
              continue;
         }}

         int block_rows = res[split_idx_for_res].size();
         // 理论上 block_rows 应该等于 mat1_block.size()
         if (!mat1_block.empty() && block_rows != mat1_block.size()) {{
              std::cerr << "Warning: Result block rows (" << block_rows << ") does not match original mat1 block rows (" << mat1_block.size()
                        << ") for split " << split_idx_for_res << ". Using result block rows for aggregation." << std::endl;
         }}


         // 检查行偏移量是否超出范围
         if (current_res_row_offset + block_rows > total_rows) {{
              std::cerr << "Error: Result block for mat1 split " << split_idx_for_res << " overflows the final result matrix rows (offset " << current_res_row_offset
                        << " + block_rows " << block_rows << " > total_rows " << total_rows << "). Stopping aggregation." << std::endl;
               // 可能需要更健壮的错误处理
               Res.assign(total_rows, std::vector<double>(total_cols, 0.0)); // 重置为0?
              return;
         }}


        // 复制数据
        for (int r = 0; r < block_rows; ++r) {{
            // Check bounds for safety before accessing Res
            if (current_res_row_offset + r < Res.size()) {{
                 std::copy(res[split_idx_for_res][r].begin(), res[split_idx_for_res][r].end(), Res[current_res_row_offset + r].begin());
            }} else {{
                 std::cerr << "Error: Row index out of bounds during copy for split " << split_idx_for_res << std::endl;
                 // Handle error, maybe break?
                 break;
            }}
        }}
        current_res_row_offset += block_rows; // 更新下一个块的起始行
        valid_result_found = true;
        split_idx_for_res++;
    }}

     // 检查是否所有行都被填充了
     if (valid_result_found && current_res_row_offset != total_rows) {{
         std::cerr << "Warning: Horizontal aggregation completed, but the number of aggregated rows ("
                   << current_res_row_offset << ") does not match the expected total rows (" << total_rows
                   << "). Result might be incomplete or contain gaps due to errors." << std::endl;
     }} else if (!valid_result_found) {{
          std::cerr << "Warning: No valid results were aggregated in horizontal mode. Result matrix might be incorrect." << std::endl;
          // Res 已经被初始化为 0 或空了
     }}

    // --- 清理工作 (GpuMultiply 已处理 Dev1, Dev2_copy), Full_Dev2_orig 已被清理
    """

    else:
         raise ValueError(f"不支持的切分模式: {split_mode}. 请选择 'vertical' 或 'horizontal'.")


    # 组合所有部分
    full_body = common_setup + split_logic + thread_creation_code + join_aggregate_code
    return full_body

def generate_mlp_cpp_file(num_gpu_chiplets, mapper, split_mode, template_cpp_path="MLP/MLP/mlp.cpp"):
    """生成包含动态ToGPU函数的完整mlp.cpp文件内容"""
    # 1. 读取模板 C++ 文件内容
    try:
        with open(template_cpp_path, 'r', encoding='utf-8') as f:
            template_code = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：找不到模板文件 {template_cpp_path}")

    # 2. 生成 ToGPU 函数的新主体
    # 创建一个临时的ChipletMapper实例以传递给generate_togu_function_body
    # 需要确保 num_gpu_chiplets > 0 才能创建
    temp_mapper_for_cpp = None
    if num_gpu_chiplets > 0:
        try:
            temp_mapper_for_cpp = ChipletMapper(num_gpu_chiplets)
        except ValueError as e:
             raise RuntimeError(f"无法创建 ChipletMapper: {e}")
    else:
        # 如果 num_gpu_chiplets 为 0，仍然调用 generate_togu_function_body
        # 它会返回一个简单的注释体，不需要 mapper
        pass

    new_togu_body = generate_togu_function_body(num_gpu_chiplets, temp_mapper_for_cpp, split_mode) # 传递 split_mode

    # 3. 替换模板中的 ToGPU 函数体
    # 使用正则表达式查找并替换 ToGPU 函数体
    # (\s*): 匹配前面的空白（包括换行符）
    # ToGPU\s*\(.*?\) : 匹配函数签名 ToGPU(...)
    # \s*\{ : 匹配函数体开始的 '{'
    # (.*?) : 非贪婪匹配函数体内容 (这是我们要替换的部分)
    # \n\}(\s*;) : 匹配函数体结束的 '}' 和可能的分号（考虑到函数可能在类定义等复杂结构中）
    # re.DOTALL 使 '.' 可以匹配换行符
    pattern = re.compile(
        r"(void\s+ToGPU\s*\(.*?)\s*\{(.*?)\n\}",
        re.DOTALL
    )

    # 在替换体中，我们保留捕获组1（函数签名），然后插入新函数体
    replacement = r"\1 {{\n{body}\n}}".format(body=new_togu_body)

    modified_code, num_replacements = pattern.subn(replacement, template_code)

    if num_replacements == 0:
        # 在替换失败时打印更多信息有助于调试
        print(f"DEBUG: Template content sample:\n{template_code[:500]}...") # 打印模板开头部分
        raise RuntimeError(f"错误：无法在模板 {template_cpp_path} 中找到或替换 ToGPU 函数体。请检查模板文件结构和正则表达式。")
    if num_replacements > 1:
        print(f"警告：在模板 {template_cpp_path} 中找到并替换了多个 ToGPU 函数体 ({num_replacements} 次)。")

    # 4. 添加文件头注释
    header_comment = f"""/*
 * @Author: auto-generated by generate_mlp_files.py
 * @Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * @Description: 自动生成的 MLP CPU 控制程序 (使用 {num_gpu_chiplets} 个 GPU 芯片, {split_mode} 切分模式)
 */
"""
    final_code = header_comment + modified_code

    return final_code


# --- YAML 配置生成逻辑 ---

def generate_mlp_yml_config(num_gpu_chiplets, mapper):
    """为MLP生成匹配的YAML配置文件"""
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)  # 控制缩进

    config = CommentedMap()

    # ========== Phase1配置 ==========
    phase1 = CommentedSeq()

    # 添加 gpgpusim 进程 (mlp_cu)
    # 只有在 num_gpu_chiplets > 0 时才添加
    if num_gpu_chiplets > 0 and mapper:
        for i, (x, y) in enumerate(mapper.coords):
            proc = CommentedMap()
            # 添加注释块
            proc.yaml_set_comment_before_after_key(
                "cmd",
                before=f"\n  # Process {i} (GPU Chiplet at ({x},{y}))"
            )
            proc["cmd"] = dq("$BENCHMARK_ROOT/bin/mlp_cu") # 使用 mlp_cu
            proc["args"] = CommentedSeq([dq(str(x)), dq(str(y))])
            proc["args"].fa.set_flow_style()
            proc["log"] = dq(f"gpgpusim.{x}.{y}.log") # 日志名保持 gpgpusim 风格？或者 mlpcu？保持一致性
            proc["is_to_stdout"] = False
            proc["clock_rate"] = 1
            proc["pre_copy"] = dq("$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*")
            phase1.append(proc)
    elif num_gpu_chiplets <= 0:
        print("警告: GPU 核心数为 0， YAML 配置中不包含 gpgpusim 进程。")

    # 添加 sniper 进程 (mlp_cpu, 运行我们生成的C++代码)
    sniper_proc = CommentedMap()
    # Sniper 进程索引取决于 GPU 数量
    sniper_proc_index = num_gpu_chiplets
    sniper_proc.yaml_set_comment_before_after_key(
        "cmd",
        before=f"\n  # Process {sniper_proc_index} (Main CPU Control at (0,0))"
    )
    # 命令需要执行我们生成的 specific C++ 可执行文件
    # 文件名将在主逻辑中根据 split_mode 动态设置
    # 这里先用一个占位符或默认值
    placeholder_executable = f"mlp_cpu_{num_gpu_chiplets}_placeholder"
    sniper_proc["cmd"] = dq("$SIMULATOR_ROOT/snipersim/run-sniper")
    sniper_proc["args"] = CommentedSeq([
        dq("--"),
        # 注意：这里的可执行文件名将在主程序中被正确设置
        dq(f"$BENCHMARK_ROOT/bin/{placeholder_executable}"), # 占位符
        dq("0"), # Sniper 进程通常在 (0,0)
        dq("0")
    ])
    sniper_proc["args"].fa.set_flow_style()
    sniper_proc["log"] = dq("sniper.0.0.log")
    sniper_proc["is_to_stdout"] = False
    sniper_proc["clock_rate"] = 1
    phase1.append(sniper_proc)

    # ========== Phase2配置 (PopNet, 通常保持不变) ==========
    phase2 = CommentedSeq()
    popnet_proc = CommentedMap()
    popnet_proc.yaml_set_comment_before_after_key(
        "cmd",
        before="\n  # Process 0 (Network Simulator)"
    )
    popnet_proc["cmd"] = dq("$SIMULATOR_ROOT/popnet_chiplet/build/popnet")
    args = [
        "-A", "2", "-c", "2", "-V", "3", "-B", "12", "-O", "12",
        "-F", "4", "-L", "1000", "-T", "10000000", "-r", "1",
        # 路径可能需要根据实际情况调整
        "-I", "../bench.txt", "-R", "0", "-D", "../delayInfo.txt", "-P"
    ]
    popnet_proc["args"] = CommentedSeq([dq(arg) for arg in args])
    popnet_proc["args"].fa.set_flow_style()
    # popnet 日志名保持一致
    popnet_proc["log"] = dq("popnet.log") # 使用 mlp.yml 中的名字
    popnet_proc["is_to_stdout"] = False
    popnet_proc["clock_rate"] = 1
    phase2.append(popnet_proc)

    # ========== 构建完整配置 ==========
    config["phase1"] = phase1
    config["phase2"] = phase2
    # 这些文件路径也可能需要调整
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

# --- 主程序逻辑 ---
if __name__ == "__main__":
    try:
        num_gpu_chiplets = int(input("请输入要使用的 GPU 芯粒数量: "))
        # 允许输入 0，表示纯 CPU 执行（尽管 ToGPU 会为空）
        if num_gpu_chiplets < 0:
            raise ValueError("GPU 芯粒数量不能为负数")

        # 新增：获取切分模式
        split_mode = 'vertical' # 默认值
        if num_gpu_chiplets > 0:
             split_mode_input = input("请选择权重矩阵 (mat1) 切分模式 ('vertical' 或 'horizontal') [默认 vertical]: ").lower().strip()
             if split_mode_input == 'horizontal':
                 split_mode = 'horizontal'
                 print("使用水平切分模式。")
             elif split_mode_input == 'vertical' or not split_mode_input:
                 split_mode = 'vertical'
                 print("使用垂直切分模式。")
             else:
                  print(f"无法识别的模式 '{split_mode_input}'，将使用默认的垂直切分模式。")
        else:
             print("GPU 数量为 0，切分模式无效。")
             split_mode = 'none' # 特殊标记

        # 1. 创建 Chiplet 映射器 (仅当 GPU > 0)
        mapper = None
        if num_gpu_chiplets > 0:
            mapper = ChipletMapper(num_gpu_chiplets)
            print(f"为 {num_gpu_chiplets} 个 GPU 芯片生成的坐标: {mapper.coords}")

        # 2. 定义模板文件路径和输出文件名/目录
        template_cpp_path = "MLP/MLP/mlp.cpp" # 确保这个路径正确
        # 文件名和目录名包含 GPU 数量和模式
        mode_suffix = split_mode if num_gpu_chiplets > 0 else 'cpu_only'
        cpp_file_name = f"mlp_{num_gpu_chiplets}_{mode_suffix}.cpp"
        yaml_file_name = f"mlp_{num_gpu_chiplets}_{mode_suffix}.yml"
        output_dir = f"mlp_generated_{num_gpu_chiplets}_gpus_{mode_suffix}"
        os.makedirs(output_dir, exist_ok=True)

        # 3. 生成 C++ 文件内容
        print(f"正在生成 {output_dir}/{cpp_file_name} ...")
        # 即使 num_gpu_chiplets=0 或 split_mode='none', 也调用生成函数
        cpp_content = generate_mlp_cpp_file(num_gpu_chiplets, mapper, split_mode, template_cpp_path)
        cpp_output_path = os.path.join(output_dir, cpp_file_name)
        with open(cpp_output_path, 'w', encoding='utf-8') as f:
            f.write(cpp_content)
        print(f"C++ 文件生成成功: {cpp_output_path}")

        # 4. 生成 YAML 配置
        print(f"正在生成 {output_dir}/{yaml_file_name} ...")
        # generate_mlp_yml_config 内部处理 num_gpu_chiplets=0 的情况
        yaml_config = generate_mlp_yml_config(num_gpu_chiplets, mapper)

        # 修改 YAML 中 sniper 的命令以包含正确的模式后缀
        # 使用与 C++ 文件名匹配的可执行文件名
        expected_executable = f"mlp_cpu_{num_gpu_chiplets}_{mode_suffix}"
        sniper_process_found = False
        if 'phase1' in yaml_config:
             for proc in yaml_config['phase1']:
                  # 通过 cmd 判断 sniper 进程
                  if isinstance(proc, dict) and 'cmd' in proc and proc['cmd'].endswith('run-sniper'):
                       if 'args' in proc and isinstance(proc['args'], CommentedSeq) and len(proc['args']) > 1:
                            proc['args'][1] = dq(f"$BENCHMARK_ROOT/bin/{expected_executable}")
                            sniper_process_found = True
                            break # 假设只有一个 sniper 进程
                       else:
                            print(f"警告：在 YAML 中找到 sniper 进程，但其 'args' 格式不符合预期，无法设置可执行文件名。")
        else:
            print("警告：生成的 YAML 配置中未找到 'phase1'，无法设置 sniper 可执行文件名。")

        if not sniper_process_found:
             print("警告：未能在生成的 YAML 中找到并更新 sniper 进程的可执行文件名。请手动检查 YAML 文件。")

        yaml_output_path = os.path.join(output_dir, yaml_file_name)
        yaml = YAML()
        yaml.indent(sequence=4, offset=2)
        with open(yaml_output_path, 'w', encoding='utf-8') as f:
             yaml.dump(yaml_config, f)
        print(f"YAML 文件生成成功: {yaml_output_path}")

        print("\n自动化脚本执行完毕！")
        print(f"请注意：您需要自行编译生成的 C++ 文件 ({cpp_output_path})")
        print(f"确保编译后的可执行文件名为 '{expected_executable}' 并放置在 $BENCHMARK_ROOT/bin/ 目录下，")
        print(f"以便与 YAML 文件 ({yaml_output_path}) 中的 sniper 进程配置相匹配。")


    except ValueError as e:
        print(f"输入错误: {str(e)}")
    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
    except RuntimeError as e:
        print(f"运行时错误: {str(e)}")
    except Exception as e:
        print(f"发生意外错误: {str(e)}")
