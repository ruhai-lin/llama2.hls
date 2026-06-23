# llama2.hls

Karpathy [tinyllama stories15M](https://huggingface.co/karpathy/tinyllamas) 在 AMD KV260（`xck26-sfvc784-2LV-c`）上的 Vitis HLS 部署。FPGA 侧为单个 `kernel_decode` megakernel，每个 token 只启动一次设备 kernel，权重与 KV cache 常驻 DDR，kernel 内完成 decode forward 与 LM head。当前 FPGA host 只支持 exact argmax（`--temp 0`），LM head 直接返回 4-byte token id，不再回传 32K logits。

工具链与板卡环境以 **Vitis / Vivado 2025.2**、官方 **KV260 base platform** 为准。

## 仓库结构

```text
.
├── README.md
├── CMakeLists.txt          # 仅 CPU 本地仿真
├── model/
│   ├── stories15M.bin      # 权重（需下载，见下）
│   ├── tokenizer.bin       # Llama2 词表（已随仓库提供）
│   └── tok512.bin          # 260K 极小词表（可选）
├── src/
│   ├── main.cpp            # host 入口
│   ├── decode.cpp / .hpp   # HLS top + host 推理逻辑
│   ├── kernel_*.cpp        # decode 内部 primitive
│   └── …
└── outputs/                # 本地构建产物（git ignore）
    ├── hls/ link/ host/ bundle/ logs/
```

`kernel_*.cpp` 由 `decode.cpp` 直接 include 进同一 HLS 翻译单元，不分别生成多个 `.xo`。这里的 `kernel_*` 只表达基础 primitive；真正的 HLS kernel 只有 `kernel_decode`。代码命名空间统一为 `llama2`。

`decode.cpp` 是 megakernel 的硬件架构层，当前按三阶段顺序调度：

1. Attention：RMSNorm、Q/K/V/O GEMV、RoPE、KV cache、softmax 与 attention value 聚合。
2. FFN：RMSNorm、W1、W3、SiLU、elementwise multiply、W2。
3. LM head：RMSNorm 后直接做 streaming argmax，不回写 32K logits。

`kernel_matmul.cpp` 保持完整 GEMV primitive 语义：`out = weight x in`。当前实现已迁移到 `runq.c` 风格 Q8_0 W8A8：FP32 input 在 kernel 内按 `group_size=32` 动态量化，权重为 int8 + FP32 scale，内部做 int8 dot、group dequant 和 FP32 output 写回。`decode.cpp` 通过单一静态 call site 顺序提交 Q/K/V/O、FFN W1/W3/W2 与 LM head tile 任务，避免 HLS 按尺寸复制 matmul module。

LM head 也走同一个 `kernel_matmul`，但按 vocab tile 计算小块 logits，再由 `decode.cpp` 中的独立 argmax consumer 处理，不回写 32K logits：

```text
for vocab_base in 0..31999 step 128:
  score_tile = tok_emb_q[vocab_base:vocab_base+128] x final_norm
  best = argmax_consume(score_tile)
```

`final_norm[288]` 作为普通 GEMV input 进入 `kernel_matmul` 的 Q8_0 input quantizer；embedding 权重与其他 linear weights 共享同一个 int8 GEMV datapath。该实现保留 exact argmax，适合 hidden dim 相同的 15M 级模型复用，扩展到 42M/110M 时主要调整 `kDim`、`kVocabSize` 与权重形状。

## 前置环境

在开发机上准备（路径按你的安装位置修改）：

| 组件 | 说明 |
|------|------|
| Vitis 2025.2 | `source /opt/xilinx/2025.2/Vitis/settings64.sh` |
| KV260 platform | `/opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm` |
| AArch64 sysroot | 例如 PetaLinux / Vitis 配套的 `xilinx-zynqmp-common`，用于交叉编译 host |

确认 platform 可用：

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
platforminfo /opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm
```

应看到 `FPGA Device: xck26`、`Board Part: xck26-sfvc784-2LV-c`。

以下命令均在**本仓库根目录**执行。

## 模型与词表

权重需自行从Karpathy仓库下载到 `model/`：

```bash
cd model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
cd ..
```

`model/tokenizer.bin` 已包含在仓库中，打包 bundle 时直接使用即可。

## HLS 综合（生成 kernel_decode.xo）

本项目只有一个 HLS kernel，top 名为 `kernel_decode`，源文件为 `src/decode.cpp`。

1. 创建目录 `outputs/hls/kernel_decode/`。
2. 在该目录下新建 `hls_config.cfg`，内容如下（器件与顶层已与 KV260 对齐）：

```ini
part=xck26-sfvc784-2LV-c

[hls]
flow_target=vitis
package.output.format=xo
package.output.syn=1
syn.top=kernel_decode
syn.file=<仓库根>/src/decode.cpp
syn.cflags=-DBUILD_DECODE_KERNEL
clock=200MHz
```

将 `syn.file=` 一行里的 `<仓库根>` 换成你本机的绝对路径（`v++` 需要绝对路径）。

3. 加载 Vitis 环境后执行综合（耗时较长，日志会写在 `outputs/logs/`）：

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
v++ -c --mode hls \
  --config outputs/hls/kernel_decode/hls_config.cfg \
  --work_dir outputs/hls/kernel_decode/work \
  2>&1 | tee outputs/logs/kernel_decode_hls.log
```

成功后得到：`outputs/hls/kernel_decode/work/kernel_decode.xo`。

## v++ Link（生成 xclbin）

将 `.xo` 与 KV260 platform 链接为比特流容器。本 megakernel 的 HLS 估算 Fmax 约 38 MHz，link 时使用 **50 MHz** 并以满足 routed timing 为准（勿直接套用 200 MHz 的 HLS `clock` 配置）。

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
mkdir -p outputs/link outputs/logs/link

v++ -l -t hw \
  --platform /opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm \
  outputs/hls/kernel_decode/work/kernel_decode.xo \
  -o outputs/link/binary_container_1.xclbin \
  --clock.default_freqhz 50000000 \
  --save-temps \
  --temp_dir outputs/link/_x \
  --log_dir outputs/logs/link \
  --report_dir outputs/link/reports \
  2>&1 | tee outputs/logs/vpp_link.log
```

输出：`outputs/link/binary_container_1.xclbin`。

## Host 交叉编译

在 ARM sysroot 下编译 `llama2_host`。将 `COMMON` 换成你的 `xilinx-zynqmp-common` 路径；`-fno-PIC -fno-PIE -no-pie` 与 `-mcmodel=large` 配合使用，避免 SDK 默认 PIC 冲突。

```bash
COMMON=/path/to/xilinx-zynqmp-common-v2022.2

source /opt/xilinx/2025.2/Vitis/settings64.sh
unset LD_LIBRARY_PATH
source "$COMMON/environment-setup-cortexa72-cortexa53-xilinx-linux"
SYSROOT="$COMMON/sysroots/cortexa72-cortexa53-xilinx-linux"
mkdir -p outputs/host outputs/logs

aarch64-xilinx-linux-g++ -Wall -Wextra -std=c++2a \
  -mcmodel=large -fno-PIC -fno-PIE -no-pie -g --sysroot="$SYSROOT" \
  -Isrc \
  src/context.cpp src/decode.cpp src/main.cpp \
  src/tensor.cpp src/vocab.cpp src/weight.cpp \
  -L"$SYSROOT/usr/lib" \
  -lxilinxopencl -lxrt_coreutil -lpthread -lrt -ldl \
  -o outputs/host/llama2_host \
  2>&1 | tee outputs/logs/host_build.log
```

## 打包 Bundle

拷到 KV260 上运行的目录结构：

```text
outputs/bundle/
├── llama2_host
├── binary_container_1.bin
├── pl.dtbo
├── shell.json
└── model/
    ├── stories15M.bin
    └── tokenizer.bin
```

```bash
PLATFORM_ROOT=/opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1
mkdir -p outputs/bundle/model

cp -f outputs/host/llama2_host outputs/bundle/llama2_host
cp -f outputs/link/binary_container_1.xclbin outputs/bundle/binary_container_1.bin
cp -f "$PLATFORM_ROOT/sw/boot/pl.dtbo" outputs/bundle/pl.dtbo
cp -f model/stories15M.bin outputs/bundle/model/stories15M.bin
cp -f model/tokenizer.bin outputs/bundle/model/tokenizer.bin
chmod +x outputs/bundle/llama2_host

cat > outputs/bundle/shell.json <<'EOF'
{
  "shell_type": "XRT_FLAT",
  "num_slots": "1"
}
EOF
```

## 部署到 KV260

将 `<kv260-ip>` 换成板卡地址（示例 `192.168.137.123`）。

```bash
KV260=ubuntu@<kv260-ip>
# ssh ubuntu@192.168.137.123
# password: ubuntu
ssh "$KV260" "rm -rf ~/Projects/llama2_bundle && mkdir -p ~/Projects/llama2_bundle/model"
scp outputs/bundle/llama2_host \
    outputs/bundle/binary_container_1.bin \
    outputs/bundle/pl.dtbo \
    outputs/bundle/shell.json \
    "$KV260":~/Projects/llama2_bundle/
scp outputs/bundle/model/stories15M.bin \
    outputs/bundle/model/tokenizer.bin \
    "$KV260":~/Projects/llama2_bundle/model/
```

在 KV260 上安装并加载 XRT app（`xmutil` 应用名为 `llama2`）：

```bash
sudo mkdir -p /lib/firmware/xilinx/llama2
sudo cp ~/Projects/llama2_bundle/binary_container_1.bin \
         ~/Projects/llama2_bundle/pl.dtbo \
         ~/Projects/llama2_bundle/shell.json \
         /lib/firmware/xilinx/llama2/
sudo xmutil unloadapp
sudo xmutil loadapp llama2
sudo xmutil listapps
```

## 上板测试

生成 16 个 token 做冒烟测试：

```bash
cd ~/Projects/llama2_bundle
./llama2_host --max_seq 16 --temp 0
```

KV260 实测参考：

| Command | Time | Speed |
|---|---:|---:|
| `./llama2_host --max_seq 16 --temp 0` | 1.42029 s | 11.2653 tok/s |
| `./llama2_host --max_seq 64 --temp 0` | 6.13650 s | 10.4294 tok/s |

测完后可恢复默认 starter app：

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## PPA 与瓶颈

本节数据来自 2025.2 HLS report、routed implementation report 与 KV260 实测。HLS cycle 用于定位相对瓶颈；板上端到端吞吐还包含 XRT kernel launch、host 打印、cache 读写和 attention 随 `pos` 增长的开销。W8A8 版本按 `runq.c` 的 Q8_0 语义工作：matmul 权重为 int8 + scale，activation 在 `kernel_matmul` 内动态量化，非 matmul primitive 仍为 FP32。

### 资源与时序

| Scope | LUT | LUT as Mem | FF/REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|---:|
| W8A8 routed `kernel_decode` | 17.97% user budget (20,015) | 3.37% (1,900) | 10.52% (23,926) | 14.58% (21) | 0% (0) | 3.45% (43) |
| W8A8 full routed design | 21.98% device (25,738) | 5.53% (3,185) | 13.15% (30,804) | 14.58% tile (21) | 0% (0) | 3.45% (43) |
| W8A8 HLS `kernel_decode` estimate | 32% device (38,280) | - | 10% (25,210) | 19% BRAM_18K (56) | 0% (0) | 3% (42) |
| HLS W8A8 `kernel_matmul` module | 7% device (9,080) | - | 1% (3,204) | 0% (0) | 0% (0) | 1% (17) |
| Prior FP32 shared GEMV routed `kernel_decode` | 20.86% user budget (23,239) | 4.06% (2,287) | 13.01% (29,590) | 14.58% (21) | 0% (0) | 9.54% (119) |
| Three-clone FP32 baseline routed `kernel_decode` | 39.86% user budget (44,399) | 7.20% (4,052) | 23.52% (53,469) | 15.28% (22) | 0% (0) | 27.00% (337) |

Timing: W8A8 routed WNS = `0.768 ns`，WHS = `0.010 ns`，50 MHz user kernel clock met。HLS 对 `kernel_decode` 的 200 MHz 估算仍报 timing negative，Estimated Fmax = `38.86 MHz`；critical path 仍主要来自 FP32 primitive，包括 Q8_0 dequant accumulation、argmax compare、RMSNorm/softmax/SiLU 等。

单实例 GEMV 复用验收：达标。`decode.cpp` 只有一个 `kernel_matmul(...)` 静态调用点，HLS hierarchy 中也只有一个 `kernel_matmul` module；没有 `288 x 288`、`768 x 288`、`288 x 768` 或 LM-head tile clone。相比 FP32 shared GEMV，W8A8 routed `kernel_decode` DSP 从 9.54% 降到 3.45%，LUT 从 20.86% 降到 17.97%，FF/REG 从 13.01% 降到 10.52%。相比三 clone FP32 baseline，DSP 从 27.00% 降到 3.45%。

### HLS Cycle Bottleneck

W8A8 版本不再产生按尺寸拆开的 matmul module，因此没有 `288 x 288`、`768 x 288`、`288 x 768` 各自的独立 HLS cycle。`kernel_matmul` 先按 Q8_0 group 动态量化 input；group max loop Final II = 1，quantize loop Final II = 1。主 row/group loop `VITIS_LOOP_58_4_VITIS_LOOP_61_5` 目标 II = 1，但实际 Final II = 2；HLS 报告原因包括同一 `gmem` 上同时读取 int8 weight 和 FP32 scale，以及 FP32 dequant partial sum。HLS max latency = 39,722 cycles。LM-head argmax consumer Final II = 1，128-row tile latency = 131 ~ 771 cycles。按 Q8 group 数估算固定 GEMV 主循环如下：

| Operator group | Rows/token | Q8 groups/row | Main-loop cycles/token |
|---|---:|---:|---:|
| Attention Q/K/V/O GEMV | 6,912 | 9 groups x II=2 | 124,416 |
| FFN W1/W3 GEMV | 9,216 | 9 groups x II=2 | 165,888 |
| FFN W2 GEMV | 1,728 | 24 groups x II=2 | 82,944 |
| LM head GEMV tiles | 32,000 | 9 groups x II=2 | 576,000 |
| LM head tile argmax | 32,000 | ~1 | 32,000 |

这个估算只覆盖固定 GEMV 主循环与 tile argmax，不包含 dynamic quantization、attention cache/softmax 的变长部分，也不包含每次 `kernel_matmul` 的 pipeline depth 和 call/control 开销。板上实测是最终判断：W8A8 在 `--max_seq 16 --temp 0` 为 11.2653 tok/s，在 `--max_seq 64 --temp 0` 为 10.4294 tok/s；FP32 shared GEMV 分别为 11.8331 tok/s 和 10.9156 tok/s。也就是说，本版主要收益是面积，吞吐因 `kernel_matmul` 仍为 II=2 且增加 dynamic quantization，较 FP32 shared GEMV 下降约 5%。

### 带宽

HLS interface 报告显示：

| Port group | Width | Burst |
|---|---:|---|
| `m_axi_gmem` | `32 -> 512` bit | max read/write burst length 16 |

Link cfgen 将所有 top-level pointer argument 映射到 KV260 base platform 的 `HP0` path。以 50 MHz 计算，单个 512-bit HLS 端口的理想接口上限是 `512/8*50MHz = 3.2 GB/s`，但实际还受共享 HP0、DDR 控制器、burst 质量和 kernel launch 开销影响。

`kernel_matmul` 当前每个 Q8 group 消费 32 个 int8 weight 和 1 个 FP32 scale。HLS interface 仍将 `m_axi_gmem` 从 32-bit widening 到 512-bit，但主 row/group loop 未达到 II=1；报告明确给出同一 `gmem` 上 weight_q 与 weight_s 多读竞争是 II 下界为 2 的原因之一。后续若继续优化 W8A8，应优先考虑把 weight scale 与 int8 weight 打包成同一连续 stream、或在 `decode.cpp` 中为 scale/weight 分 bundle，而不是继续只调 unroll。

仍存在若干非理想访问：KV cache 的跨 head/pos 访问有 stride incompatible，RoPE sin/cos 小表访问有 pattern/widen fail；共享 `m_axi_gmem` 还让 RoPE 的 sin/cos 双读和 KV cache 双写部分 pipeline 因 bus resource 限制到 II=2。这些 stall 不来自 shared linear engine 的主循环。

## 参考资料

- [AMD KV260 platform 教程](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-Platform-Creation/Custom-Platform-Creation-Tutorial-on-MPSoC)
- [turingmotors/swan](https://zenn.dev/turing_motors/articles/82505880d27d65)
- [本系列 Vitis 2025 部署笔记](https://blog.csdn.net/Bird_Boss/article/details/151792370)
