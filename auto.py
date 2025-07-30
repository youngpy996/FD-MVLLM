# 2.py
import subprocess
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

"""
E:\young\online\DeepSeek-R1-Distill-Llama-8B      4096      deepseek    True
DeepSeek-R1-Distill-Qwen-7B                       4096      deepseek    True
Qwen2.5-7B-Instruct                               3584      LLAMA       True
Meta-Llama-3-8B                                   4096      LLAMA       True
bert-base-uncased                                 768       BERT        False
gpt2                                              768       GPT2        False
"""
# 定义三组参数的对应关系
model_roots = [
    r'E:\young\online\DeepSeek-R1-Distill-Llama-8B',
    r'E:\young\online\DeepSeek-R1-Distill-Qwen-7B',
    r'E:\young\online\Qwen2.5-7B-Instruct',
    r'E:\young\online\Meta-Llama-3-8B',
    # r'E:\young\online\bert-base-uncased'
    # r'E:\young\online\gpt2'
]

models = [
    'deepseek',
    'deepseek',
    'LLAMA',
    'LLAMA',
    # 'BERT',
    # 'GPT2'
]

model_dims = [
    4096,
    4096,
    3584,
    4096,
    # 768,
    # 768
]

llm_lora = [
    True,
    True,
    True,
    True,
    # False,
    # False,
]

# 确保三个列表长度一致
assert len(model_roots) == len(models) == len(model_dims), "参数列表长度必须相同"

# 遍历所有参数组合
for model_root, model, dim, lora in zip(model_roots, models, model_dims, llm_lora):
    # 构造命令行参数
    cmd = [
        sys.executable,  # 使用当前Python解释器
        'testV2.py',
        '--llm_model_root', model_root,
        '--llm_model', model,
        '--llm_dim', str(dim),  # 转换为字符串类型
        '--llm_lora', str(lora).lower(),
        '--task_name', 'classification',
        '--is_training', str(1),
        '--model_id', 'ECL_512_96',
        '--model', 'test',
        '--data', 'ECL',
        '--model_comment', 'none',
    ]

    # 打印当前运行的参数组合
    print(f"\n正在运行参数组合:")
    print(f"Model Root: {model_root}")
    print(f"Model Name: {model}")
    print(f"Model Dim: {dim}"),
    print(f"Model lora: {lora}")

    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    # 打印输出结果
    print("\n标准输出：")
    print(result.stdout)
    if result.stderr:
        print("错误信息：")
        print(result.stderr)
    print("-" * 50)
