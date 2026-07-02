# llama2.hls

Karpathy [tinyllama stories15M](https://huggingface.co/karpathy/tinyllamas) 在 AMD KV260（`xck26-sfvc784-2LV-c`）上的 Vitis HLS 部署。FPGA 侧为单个 `decode` kernel，每个 token 只启动一次设备 kernel，权重与 KV cache 常驻 DDR，kernel 内完成 decode forward 与 LM head。当前 FPGA host 只支持 exact argmax（`--temp 0`），LM head 直接返回 4-byte token id，不再回传 32K logits。

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
│   └── …
└── outputs/                # 本地构建产物（git ignore）
    ├── hls/ link/ host/ bundle/ logs/
```

HLS 核心集中在 `decode.cpp`。真正的 HLS top 只有 `decode`，内部按专用数据流组织为：

```text
decode
  for layer:
    attn
    ffn
  lm_head
```

小 primitive 不再拆成独立源文件，而是作为 `decode.cpp` 内的 `static` helper 存在。代码命名空间统一为 `llama2`。

`decode.cpp` 是 megakernel 的硬件架构层，当前按三阶段顺序调度：

1. Attention：RMSNorm、Q/K/V/O GEMV、RoPE、KV cache、softmax 与 attention value 聚合。
2. FFN：RMSNorm、W1、W3、SiLU、elementwise multiply、W2。
3. LM head：RMSNorm 后直接做 streaming argmax，不回写 32K logits。

`decode.cpp` 内的 `matmul<OutSize, InSize>` 是固定尺寸 FP32 GEMV helper，内部完成输入缓存、512-bit weight load、16-lane MAC accumulation、reduction 和 `out[row]` 写回。Q/K/V/O、FFN W1/W3/W2 通过编译期尺寸实例化，不走 runtime mode。

LM head 不走通用 writeback 路径，专门在 `lm_head` 中流式扫描 embedding rows 并维护 argmax：

```text
for row in 0..31999:
  score = tok_emb_table[row] x final_norm
  best = max(best, score)
```

`final_norm[288]` 只在 LM head 内缓存一次，embedding 权重按 512-bit word 顺序读取。该实现保留 exact argmax，适合 hidden dim 相同的 15M 级模型复用，扩展到 42M/110M 时主要调整 `kDim`、`kVocabSize` 与权重形状。

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

## HLS 综合（生成 decode.xo）

本项目只有一个 HLS kernel，top 名为 `decode`，源文件为 `src/decode.cpp`。

1. 创建目录 `outputs/hls/decode/`。
2. 在该目录下新建 `hls_config.cfg`，内容如下（器件与顶层已与 KV260 对齐）：

```ini
part=xck26-sfvc784-2LV-c

[hls]
flow_target=vitis
package.output.format=xo
package.output.syn=1
syn.top=decode
syn.file=<仓库根>/src/decode.cpp
syn.cflags=-DBUILD_DECODE_KERNEL
clock=150MHz
```

将 `syn.file=` 一行里的 `<仓库根>` 换成你本机的绝对路径（`v++` 需要绝对路径）。

3. 加载 Vitis 环境后执行综合（耗时较长，日志会写在 `outputs/logs/`）：

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
v++ -c --mode hls \
  --config outputs/hls/decode/hls_config.cfg \
  --work_dir outputs/hls/decode/work \
  2>&1 | tee outputs/logs/decode_hls.log
```

成功后得到：`outputs/hls/decode/work/decode.xo`。

## v++ Link（生成 xclbin）

将 `.xo` 与 KV260 platform 链接为比特流容器。当前目标使用 platform 默认方向的 **150 MHz**，最终以 routed timing 和板上吞吐为准。

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
mkdir -p outputs/link outputs/logs/link

v++ -l -t hw \
  --platform /opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm \
  outputs/hls/decode/work/decode.xo \
  -o outputs/link/binary_container_1.xclbin \
  --clock.default_freqhz 150000000 \
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
从 WSL shell 执行下面的 `ssh`/`scp`；KV260 默认没有 Windows 侧的 SSH key。

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
| `./llama2_host --max_seq 16 --temp 0` | 1.35214 s | 11.8331 tok/s |
| `./llama2_host --max_seq 64 --temp 0` | 5.86317 s | 10.9156 tok/s |

测完后可恢复默认 starter app：

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## PPA 与瓶颈

本节数据来自 2025.2 HLS report 与 routed implementation report。当前 `decode -> attn/ffn/lm_head` 重构已经通过 HLS 并生成 `decode.xo`，但 150 MHz link 未过 routed timing，因此没有新的可烧写 bitstream，也没有本轮板上吞吐。旧 KV260 实测保留作重构前 baseline。HLS cycle 用于定位相对瓶颈；板上端到端吞吐还包含 XRT kernel launch、host 打印、cache 读写和 attention 随 `pos` 增长的开销。

### 资源与时序

| Scope | LUT | LUT as Mem | FF/REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|---:|
| Current fixed-stage HLS estimate | 56% device (65,591) | - | 22% (53,367) | 25% BRAM_18K (72) | 0% (0) | 27% (337) |
| Current fixed-stage routed `decode` (timing failed) | 40.26% user budget (44,858) | 4.72% (2,659) | 21.49% (48,863) | 14.58% (21) | 0% (0) | 27.00% (337) |
| Current fixed-stage full routed design (timing failed) | 43.17% device (50,560) | 6.84% (3,942) | 23.77% (55,677) | 14.58% tile (21) | 0% (0) | 27.00% (337) |
| Legacy shared GEMV routed `decode` | 20.86% user budget (23,239) | 4.06% (2,287) | 13.01% (29,590) | 14.58% (21) | 0% (0) | 9.54% (119) |
| Shared GEMV full routed design | 24.74% device (28,970) | 6.20% (3,572) | 15.57% (36,466) | 14.93% tile (21.5) | 0% (0) | 9.54% (119) |
| Legacy shared GEMV HLS estimate | 31% device (37,337) | - | 13% (30,787) | 25% BRAM_18K (73) | 0% (0) | 9% (118) |
| Legacy shared matmul module | 8% device (9,844) | - | 4% (10,458) | 5% BRAM_18K (16) | 0% (0) | 7% (93) |
| Prior inline shared path routed `decode` | 18.97% user budget (21,129) | 3.49% (1,964) | 11.62% (26,419) | 15.28% (22) | 0% (0) | 8.33% (104) |
| Three-clone baseline routed `decode` | 39.86% user budget (44,399) | 7.20% (4,052) | 23.52% (53,469) | 15.28% (22) | 0% (0) | 27.00% (337) |
| Three-clone baseline full routed design | 42.81% device (50,136) | 9.27% (5,337) | 25.76% (60,345) | 15.28% tile (22) | 0% (0) | 27.00% (337) |

Timing: current fixed-stage HLS 目标为 150 MHz，estimated period = `26.182 ns`。150 MHz link routed 后 WNS = `-3.772 ns`，TNS = `-21888.645 ns`，setup failing endpoints = `7439`，critical path 落在 `fadd_32ns_32ns_32_2_full_dsp` 内。legacy shared GEMV routed WNS = `0.662 ns`，WHS = `0.010 ns`，50 MHz user kernel clock met。

旧单实例 GEMV 复用验收：达标。旧 `decode.cpp` 只有一个共享 matmul 静态调用点，HLS hierarchy 中也只有一个 shared matmul module；没有 `288 x 288`、`768 x 288`、`288 x 768` 或 LM-head tile clone。相比三 clone baseline，routed `decode` DSP 从 27.00% 降到 9.54%，LUT 从 39.86% 降到 20.86%，FF/REG 从 23.52% 降到 13.01%。当前结构转为固定尺寸 `matmul<OutSize, InSize>` 与专用 `lm_head` 后，代码形态更干净，但资源回到三 clone 量级，下一步要先处理 FP32 accumulation 的时序。

### HLS Cycle Bottleneck

旧 shared GEMV 版本不再产生按尺寸拆开的 matmul module，因此没有 `288 x 288`、`768 x 288`、`288 x 768` 各自的独立 HLS cycle。旧 shared matmul 的 input-cache loop Final II = 1；GEMV row/tile loop `VITIS_LOOP_30_2_VITIS_LOOP_37_4` Final II = 1，HLS max tripcount 按 `768 x 768` 报告为 36,864，latency = 36,972 cycles。LM-head argmax consumer `VITIS_LOOP_232_10` Final II = 1，128-row tile latency = 129 cycles。按 16-wide tile 数估算固定 GEMV 主循环如下：

| Operator group | Rows/token | 16-wide tiles/row | Main-loop cycles/token |
|---|---:|---:|---:|
| Attention Q/K/V/O GEMV | 6,912 | 18 | 124,416 |
| FFN W1/W3 GEMV | 9,216 | 18 | 165,888 |
| FFN W2 GEMV | 1,728 | 48 | 82,944 |
| LM head GEMV tiles | 32,000 | 18 | 576,000 |
| LM head tile argmax | 32,000 | ~1 | 32,000 |

这个估算只覆盖固定 GEMV 主循环与 tile argmax，不包含 attention cache/softmax 的变长部分，也不包含每次 shared matmul 的 input-cache loop、pipeline depth 和 call/control 开销。板上实测是最终判断：legacy shared GEMV 在 `--max_seq 16 --temp 0` 为 11.8331 tok/s，在 `--max_seq 64 --temp 0` 为 10.9156 tok/s；三 clone baseline 分别为 12.2114 tok/s 和 11.2376 tok/s。

### 带宽

HLS interface 报告显示：

| Port group | Width | Burst |
|---|---:|---|
| `m_axi_gmem` | `32 -> 512` bit | max read/write burst length 16 |

Link cfgen 将所有 top-level pointer argument 映射到 KV260 base platform 的 `HP0` path。以 150 MHz 计算，单个 512-bit HLS 端口的理想接口上限是 `512/8*150MHz = 9.6 GB/s`，但实际还受共享 HP0、DDR 控制器、burst 质量和 kernel launch 开销影响。

`matmul<OutSize, InSize>` 和 `lm_head` 使用显式 512-bit vector load 消费 16 个 FP32 weight。未使用 vector load 的初版会让 inner loop 退化为 Final II = 16，原因是同一 `gmem` 上 16 次 32-bit bus read 冲突；当前结构仍保留 512-bit vector load。HLS burst diagnostics 对普通数组访问只列出 `tok_emb_table`、KV cache、RMSNorm、RoPE 等变量，未把 vector-cast weight load 单独展开成 `ffn_w*`/`attn_w*` 的 widened burst 条目；因此这里确认的是 `m_axi_gmem` 端口宽度与 vector load 结构，具体 II 和吞吐需要看本轮 HLS/上板结果。

仍存在若干非理想访问：KV cache 的跨 head/pos 访问有 stride incompatible，RoPE sin/cos 小表访问有 pattern/widen fail；共享 `m_axi_gmem` 还让 RoPE 的 sin/cos 双读和 KV cache 双写部分 pipeline 因 bus resource 限制到 II=2。这些 stall 需要在本轮 HLS 报告和板上吞吐中重新确认。

## 参考资料

- [AMD KV260 platform 教程](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-Platform-Creation/Custom-Platform-Creation-Tutorial-on-MPSoC)
- [turingmotors/swan](https://zenn.dev/turing_motors/articles/82505880d27d65)
- [本系列 Vitis 2025 部署笔记](https://blog.csdn.net/Bird_Boss/article/details/151792370)
