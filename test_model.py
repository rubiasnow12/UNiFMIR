import torch
import argparse
import sys

# ----------------------------------------------------------------
# 1. 尝试从 model.swinir 导入你修改后的 make_model
#    (我们假设此脚本在 UNiFMIR-main 根目录运行)
# ----------------------------------------------------------------
print("正在导入 'make_model' 从 'model.dinoir_v3'...")
try:
    # 这里的 'model' 是文件夹名, 'swinir' 是文件名, 'make_model' 是函数名
    from model.dinoir_v3 import make_model
    print("...导入成功！")
except ImportError as e:
    print(f"\n[导入失败]: {e}")
    print("错误：无法导入 'make_model'。")
    print("请确保您是在 'UNiFMIR-main' 目录下运行此脚本,")
    print("并且 'model/dinoir_v3' 文件存在。")
    sys.exit(1) # 退出脚本
except NameError as e:
    print(f"\n[导入失败]: 变量未定义 (NameError) - {e}")
    print("错误：在 dinoir_v3.py 文件顶层（导入时）就遇到了未定义的变量。")
    print("请检查：您是否已将 Mlp, SwiGLUFFN 等类的定义粘贴到文件顶部？")
    print("       并且，ffn_layer_dict 等字典是否定义在 *所有类* 之后？")
    sys.exit(1) # 退出脚本
except Exception as e:
    print(f"\n[导入失败]: 发生未知错误: {type(e).__name__} - {e}")
    print("请检查 'model/dinoir_v3.py' 文件中的代码是否有拼写错误或其他语法错误。")
    sys.exit(1) # 退出脚本

# ----------------------------------------------------------------
# 2. 创建一个模拟的 'args' 对象
#    make_model(args) 
#    至少需要 args.scale 和 args.inputchannel
# ----------------------------------------------------------------
print("\n正在创建模拟的 'args' 对象...")

# 使用 argparse.Namespace 来创建一个可以添加属性的空对象
args = argparse.Namespace()

# 模拟 mainSR.py 中的参数
# 'scale' 在代码中会被处理成一个列表
args.scale = [4]  # 假设放大倍数 (upscale) 为 4

# 'inputchannel' 对应 'n_colors' 参数
args.inputchannel = 1 # 假设输入通道 (in_chans) 为 1

print(f" - 设置 args.scale = {args.scale}")
print(f" - 设置 args.inputchannel = {args.inputchannel}")

# ----------------------------------------------------------------
# 3. 运行模型实例化测试 (核心步骤)
# ----------------------------------------------------------------
print("\n--- 开始模型实例化测试 ---")

try:
    # 这是测试的核心：调用 make_model
    model = make_model(args)

    # 如果成功:
    print("\n--- \033[92m测试成功!\033[0m ---") # 绿色成功提示
    print("模型已成功创建实例。")
    print("这意味着 __init__ 方法中的所有层都已正确初始化。")

    # 打印模型结构，您可以检查是否有 DINOv3 的 'blocks' 和 'rope_embed'
    print("\n模型结构摘要 (部分):")
    print(model) 

    print("\n--- 开始前向传播测试 ---")
    H, W = 64, 64
    print(f"创建模拟输入张量: (1, {args.inputchannel}, {H}, {W})")
        
    # 尝试将其移动到 GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将模型和张量移动到: {device}")
        
    try:
        model = model.to(device)
        dummy_input = torch.randn(1, args.inputchannel, H, W).to(device)
        
        # 设置模型为评估模式 (关闭 dropout 等)
        model.eval()
        
        # 不计算梯度，以节省内存
        with torch.no_grad():
            # 这是测试的核心：调用 model.forward(input)
            output = model(dummy_input)
        
        # 如果成功:
        print(f"\n前向传播完成。")
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        
        # 验证输出形状
        expected_H = H * args.scale[0]
        expected_W = W * args.scale[0]
        if output.shape == (1, 1, expected_H, expected_W):
            print(f"--- \033[92m前向传播测试成功!\033[0m ---")
            print(f"输出形状 (1, 1, {expected_H}, {expected_W}) 符合预期。")
        else:
            print(f"--- \033[93m前向传播测试警告\033[0m ---") # 黄色警告
            print(f"输出形状不符合预期! 预期为 (1, 1, {expected_H}, {expected_W})。")
    except RuntimeError as e:
        print(f"\n--- \033[91m前向传播测试失败: 运行时错误 (RuntimeError)\033[0m ---")
        print(f"错误详情: {e}")
        print("\n这通常意味着张量维度不匹配 (dimension mismatch)。")
        print("请仔细检查：")
        print(" 1. 'forward_features' 中 'patch_embed' 和 'patch_unembed' 的操作是否正确？")
        print(" 2. DINOv3 'SelfAttentionBlock' 的 'forward' 方法是否正确？")
        print(" 3. 'conv_after_body' 或上采样模块的输入/输出通道是否正确？")
    
    except Exception as e:
        print(f"\n--- \033[91m前向传播测试失败: 发生未知错误\033[0m ---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")


except AttributeError as e:
    print(f"\n--- \033[91m测试失败: 属性错误 (AttributeError)\033[0m ---") # 红色失败提示
    print(f"错误详情: {e}")
    print("\n这通常意味着代码中引用了不存在的变量、函数或参数。")
    print("请仔细检查：")
    print(" 1. 所有的DINOv3依赖类是否都已粘贴？")
    print(" 2. __init__ 或 forward_features 中是否有拼写错误？")
    print(" 3. 检查 'self.xxx' 是否写错？")

except TypeError as e:
    print(f"\n--- \033[91m测试失败: 类型错误 (TypeError)\033[0m ---") # 红色失败提示
    print(f"错误详情: {e}")
    print("\n这通常意味着调用函数时参数不匹配。")
    print("请仔细检查：")
    print(" 1. 您修改的 swinir.__init__ 签名中的 DINOv3 参数是否都有默认值？")
    print(f" 2. 检查 make_model 调用 swinir(...) 时是否提供了必需的参数？(我们只传了 upscale 和 in_chans)")

except Exception as e:
    print(f"\n--- \033[91m测试失败: 发生未知错误\033[0m ---") # 红色失败提示
    print(f"错误类型: {type(e).__name__}")
    print(f"错误详情: {e}")

print("\n--- 实例化测试结束 ---")