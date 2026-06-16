import torch
import time
import io
import numpy as np
import PIL.Image
from torchvision.io import decode_jpeg, ImageReadMode
import sys

print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('=' * 70)

# =====================================================================
# GPU 利用率检测
# =====================================================================
def check_gpu_utilization():
    """检测 GPU 是否真正被利用（显存、CUDA kernel 活动等）"""
    print('\n[GPU 利用率检测]')

    # 1. 检查 nvJPEG 是否走 GPU 路径
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    # 生成测试图片
    rng = np.random.RandomState(0)
    img_arr = rng.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    img = PIL.Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    t_test = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)

    # GPU 解码
    out_test = decode_jpeg(t_test, mode=ImageReadMode.RGB, device='cuda')
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    print(f'  解码前显存占用: {mem_before / 1024:.1f} KB')
    print(f'  解码后显存占用: {mem_after / 1024:.1f} KB')
    print(f'  峰值显存占用:   {mem_peak / 1024:.1f} KB')
    print(f'  解码输出设备:   {out_test.device}')
    print(f'  解码输出 shape: {tuple(out_test.shape)}')

    # 预期显存增量 = C*H*W bytes (uint8)
    expected_bytes = 3 * 720 * 1280
    actual_increase = mem_after - mem_before
    print(f'  预期显存增量:   {expected_bytes / 1024:.1f} KB (3x720x1280 uint8)')
    print(f'  实际显存增量:   {actual_increase / 1024:.1f} KB')

    if out_test.device.type == 'cuda' and actual_increase > 0:
        print('  ✓ 确认: GPU 显存有分配，解码结果在 GPU 上，nvJPEG 路径生效')
    else:
        print('  ✗ 警告: GPU 可能未真正参与解码!')

    # 2. CUDA events 精确计时（排除 CPU 开销）
    print('\n  [CUDA Events 精确计时 - 720p x 50次]')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(5):
        _ = decode_jpeg(t_test, mode=ImageReadMode.RGB, device='cuda')
    torch.cuda.synchronize()

    cnt = 50
    start_event.record()
    for _ in range(cnt):
        _ = decode_jpeg(t_test, mode=ImageReadMode.RGB, device='cuda')
    end_event.record()
    torch.cuda.synchronize()
    cuda_elapsed = start_event.elapsed_time(end_event)  # ms
    print(f'  CUDA kernel 总耗时: {cuda_elapsed:.2f} ms ({cnt}次)')
    print(f'  CUDA kernel 每张:   {cuda_elapsed / cnt:.3f} ms')

    # wall-clock 对比
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cnt):
        _ = decode_jpeg(t_test, mode=ImageReadMode.RGB, device='cuda')
    torch.cuda.synchronize()
    wall_elapsed = (time.perf_counter() - t0) * 1000
    print(f'  Wall-clock 总耗时:  {wall_elapsed:.2f} ms ({cnt}次)')
    print(f'  Wall-clock 每张:    {wall_elapsed / cnt:.3f} ms')

    overhead = wall_elapsed - cuda_elapsed
    print(f'  CPU侧开销(launch等): {overhead:.2f} ms ({overhead/wall_elapsed*100:.1f}%)')

    if cuda_elapsed > 0.1:
        print('  ✓ 确认: CUDA kernel 有实际执行时间，GPU 确实在工作')
    else:
        print('  ✗ 警告: CUDA kernel 时间接近零，可能回退到 CPU 路径')

    # 3. 检查 torch 编译时是否支持 nvJPEG
    print('\n  [torchvision 编译信息]')
    import torchvision
    print(f'  torchvision 版本: {torchvision.__version__}')
    # 尝试检测 nvjpeg 支持
    has_nvjpeg = hasattr(torchvision.io, 'decode_jpeg')
    print(f'  decode_jpeg 可用: {has_nvjpeg}')
    try:
        # 如果不支持 cuda device 会抛异常
        _ = decode_jpeg(t_test, mode=ImageReadMode.RGB, device='cuda')
        print('  decode_jpeg(device="cuda") 调用: ✓ 成功')
    except Exception as e:
        print(f'  decode_jpeg(device="cuda") 调用: ✗ 失败 - {e}')

    print('=' * 70)


check_gpu_utilization()
# sys.exit(-1)

N = 100

# 不同分辨率配置: (名称, 宽, 高)
RESOLUTIONS = [
    ('224x224',   224,  224),
    ('480p',      640,  480),
    ('720p',     1280,  720),
    ('1080p',    1920, 1080),
]


def generate_jpeg(width, height, quality=85):
    """生成指定分辨率的随机 JPEG 图片，返回 uint8 tensor"""
    rng = np.random.RandomState(42)
    img_array = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = PIL.Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    jpeg_bytes = buf.getvalue()
    t = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)
    return t, len(jpeg_bytes)


def benchmark_decode(t, device, n=N, warmup=5):
    """对指定 device 跑 n 次解码，返回平均耗时(ms)"""
    for _ in range(warmup):
        _ = decode_jpeg(t, mode=ImageReadMode.RGB, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        out = decode_jpeg(t, mode=ImageReadMode.RGB, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1000, out  # avg ms, last output


def benchmark_cpu_then_cuda(t, n=N, warmup=5):
    """CPU 解码 + .cuda() 传输"""
    for _ in range(warmup):
        _ = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu').cuda()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        out = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu').cuda()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1000, out


# --- 打印表头 ---
header = (f'{"分辨率":<12} {"JPEG大小":>10} '
          f'{"GPU(ms)":>9} {"CPU(ms)":>9} {"CPU+cuda(ms)":>13} '
          f'{"GPU/CPU":>8} {"GPU/传输":>9}')
print(header)
print('-' * 70)

results = []

for name, w, h in RESOLUTIONS:
    t, jpeg_size = generate_jpeg(w, h)

    gpu_avg, out_gpu = benchmark_decode(t, 'cuda')
    cpu_avg, out_cpu = benchmark_decode(t, 'cpu')
    transfer_avg, _ = benchmark_cpu_then_cuda(t)

    speedup_cpu = cpu_avg / gpu_avg
    speedup_transfer = transfer_avg / gpu_avg

    # 像素差异
    diff = (out_gpu.cpu().float() - out_cpu.float()).abs()
    max_diff = diff.max().item()

    size_str = f'{jpeg_size/1024:.1f}KB'
    print(f'{name:<12} {size_str:>10} '
          f'{gpu_avg:>8.3f}  {cpu_avg:>8.3f}  {transfer_avg:>12.3f}  '
          f'{speedup_cpu:>7.2f}x {speedup_transfer:>8.2f}x')

    results.append({
        'name': name, 'w': w, 'h': h,
        'gpu_avg': gpu_avg, 'cpu_avg': cpu_avg,
        'transfer_avg': transfer_avg,
        'speedup_cpu': speedup_cpu,
        'speedup_transfer': speedup_transfer,
        'max_pixel_diff': max_diff,
    })

print('-' * 70)
print(f'(每个分辨率跑 {N} 次取平均, warmup 5 次)')
print(f'(GPU vs CPU 像素最大差异均 <= {max(r["max_pixel_diff"] for r in results):.0f}，'
      f'属于不同解码器实现的正常差异)')

# =====================================================================
# 批量解码对比 (batch decode)
# =====================================================================
print('\n')
print('=' * 70)
print('批量解码对比 (decode_jpeg 接收 list of tensors)')
print('=' * 70)

BATCH_SIZES = [16, 32]


def benchmark_batch_gpu(t, batch_size, n=N, warmup=5):
    """GPU 批量解码: 一次传入 batch_size 张"""
    batch = [t] * batch_size
    for _ in range(warmup):
        _ = decode_jpeg(batch, mode=ImageReadMode.RGB, device='cuda')
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        outs = decode_jpeg(batch, mode=ImageReadMode.RGB, device='cuda')
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # 返回: 每批总耗时(ms), 每张平均耗时(ms)
    total_per_batch = elapsed / n * 1000
    per_image = total_per_batch / batch_size
    return total_per_batch, per_image


def benchmark_batch_cpu(t, batch_size, n=N, warmup=5):
    """CPU 逐张解码 batch_size 张（模拟串行）"""
    for _ in range(warmup):
        for _ in range(batch_size):
            _ = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu')

    start = time.perf_counter()
    for _ in range(n):
        for _ in range(batch_size):
            _ = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu')
    elapsed = time.perf_counter() - start
    total_per_batch = elapsed / n * 1000
    per_image = total_per_batch / batch_size
    return total_per_batch, per_image


def benchmark_batch_cpu_then_cuda(t, batch_size, n=N, warmup=5):
    """CPU 逐张解码 + cuda() 传输，模拟常规 pipeline"""
    for _ in range(warmup):
        for _ in range(batch_size):
            _ = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu').cuda()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        for _ in range(batch_size):
            _ = decode_jpeg(t, mode=ImageReadMode.RGB, device='cpu').cuda()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_per_batch = elapsed / n * 1000
    per_image = total_per_batch / batch_size
    return total_per_batch, per_image


for bs in BATCH_SIZES:
    print(f'\n--- batch_size = {bs} ---')
    header_batch = (f'{"分辨率":<12} '
                    f'{"GPU批量(ms)":>12} {"GPU每张(ms)":>12} '
                    f'{"CPU串行(ms)":>12} {"CPU每张(ms)":>12} '
                    f'{"CPU+cuda(ms)":>13} '
                    f'{"批量/串行":>10} {"批量/传输":>10}')
    print(header_batch)
    print('-' * 95)

    for name, w, h in RESOLUTIONS:
        t, jpeg_size = generate_jpeg(w, h)

        gpu_total, gpu_per = benchmark_batch_gpu(t, bs)
        cpu_total, cpu_per = benchmark_batch_cpu(t, bs)
        transfer_total, transfer_per = benchmark_batch_cpu_then_cuda(t, bs)

        speedup_vs_cpu = cpu_total / gpu_total
        speedup_vs_transfer = transfer_total / gpu_total

        print(f'{name:<12} '
              f'{gpu_total:>11.2f}  {gpu_per:>11.3f}  '
              f'{cpu_total:>11.2f}  {cpu_per:>11.3f}  '
              f'{transfer_total:>12.2f}  '
              f'{speedup_vs_cpu:>9.2f}x {speedup_vs_transfer:>9.2f}x')

    print('-' * 95)
    print(f'(每批 {bs} 张, 跑 {N} 次取平均)')
