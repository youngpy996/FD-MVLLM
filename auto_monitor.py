# 2.py
import subprocess
import sys
import time
from datetime import datetime, timedelta
import pynvml
import os

"""
E:\young\online\DeepSeek-R1-Distill-Llama-8B      4096      deepseek    True
DeepSeek-R1-Distill-Qwen-7B                       3584      deepseek    True
Qwen2.5-7B-Instruct                               3584      deepseek    True
Meta-Llama-3-8B                                   4096      LLAMA       True
bert-base-uncased                                 768       BERT        False
gpt2                                              768       GPT2        False
"""


# # 在文件开头添加CUDA调试环境变量
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 关键修复1：同步错误报告
# os.environ['PYTHONFAULTHANDLER'] = '1'    # 增强错误跟踪


def get_gpu_utilization():
    """获取第一个GPU的使用率"""
    if not pynvml:
        print("警告：pynvml未安装，无法监测GPU使用率")
        return 0

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return util.gpu
    except Exception as e:
        print(f"获取GPU使用率失败：{str(e)}")
        return 100  # 出现错误时视为高负载


# 定义参数组合
model_roots = [
    r'E:\young\online\DeepSeek-R1-Distill-Llama-8B',
    r'E:\young\online\DeepSeek-R1-Distill-Qwen-7B',
    r'E:\young\online\Qwen2.5-7B-Instruct',
    r'E:\young\online\Meta-Llama-3-8B',
    r'E:\young\online\bert-base-uncased',
    r'E:\young\online\gpt2'
]

models = [
    'deepseek',
    'deepseek',
    'deepseek',
    'LLAMA',
    'BERT',
    'GPT2'
]

model_dims = [
    4096,
    3584,
    3584,
    4096,
    768,
    768
]

llm_lora = [
    True,
    True,
    True,
    True,
    False,
    False
]

# 确保所有参数列表长度一致
assert len(model_roots) == len(models) == len(model_dims) == len(llm_lora)

# GPU监测参数
CHECK_INTERVAL = 10  # 检测间隔（秒）
REQUIRED_IDLE_DURATION = 10  # 要求持续空闲时间（秒）
last_high_usage_time = None  # 最后高负载时间


def run_experiments():
    """运行所有实验组合"""
    for model_root, model, dim, lora in zip(model_roots, models, model_dims, llm_lora):
        cmd = [
            sys.executable,
            'testV2.py',
            '--llm_model_root', model_root,
            '--llm_model', model,
            '--llm_dim', str(dim),
            '--llm_lora', str(lora).lower(),
            '--task_name', 'classification',
            '--is_training', '1',
            '--model_id', 'ECL_512_96',
            '--model', 'test',
            '--data', 'ECL',
            '--model_comment', 'none',
        ]

        print(f"\n正在运行参数组合：")
        print(f"Model Root: {model_root}")
        print(f"Model Name: {model}")
        print(f"Model Dim: {dim}")
        print(f"LoRA Enable: {lora}")

        # result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        # result = subprocess.run(cmd, capture_output=True,
        #                         # text=True,
        #                         check=True,
        #                         timeout=600, encoding='utf-8')  # 设置合理超时
        result = subprocess.run(cmd, timeout=600)
        print("\n标准输出：")
        print(result.stdout)
        if result.stderr:
            print("错误信息：")
        print(result.stderr)
        print("-" * 50)

        # 主监控循环


while True:
    try:
        utilization = get_gpu_utilization()
        now = datetime.now()

        # 更新最后高负载时间
        if utilization >= 10:
            last_high_usage_time = now
            status = "高负载" if utilization >= 10 else "空闲"
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] GPU使用率：{utilization}% ({status})")

        # 检查空闲持续时间
        if last_high_usage_time:
            idle_duration = (now - last_high_usage_time).total_seconds()
            if idle_duration >= REQUIRED_IDLE_DURATION:
                print(f"\n检测到GPU持续空闲超过{REQUIRED_IDLE_DURATION // 60}分钟，开始运行实验...")
                run_experiments()
                last_high_usage_time = datetime.now()  # 重置计时
        else:
            last_high_usage_time = now  # 初始化计时

        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n监控已停止")
        break
    except Exception as e:
        print(f"监控出错：{str(e)}")
        time.sleep(CHECK_INTERVAL)
