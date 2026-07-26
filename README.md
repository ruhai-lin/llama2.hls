# llama2.hls

Karpathy [tinyllama stories15M](https://huggingface.co/karpathy/tinyllamas) 在 AMD KV260（`xck26-sfvc784-2LV-c`）上的 Vitis HLS 部署。FPGA 侧为单个 `decode` kernel，每个 token 只启动一次设备 kernel，Q8_0 权重与 FP32 KV cache 常驻 DDR，kernel 内完成 W8A8 decode forward 与 LM head。FPGA host 支持 `generate`、`chat` 和 `server` 三种模式，均使用 exact argmax（`--temp 0`）；LM head 直接返回 4-byte token id，不回传 32K logits。

工具链与板卡环境以 **Vitis / Vivado 2025.2**、官方 **KV260 base platform** 为准。

## 仓库结构

```text
.
├── README.md
├── CMakeLists.txt          # 仅 CPU 本地仿真
├── model/
│   ├── stories15M_q8.bin   # llama2.c v2 Q8_0 权重
│   ├── tokenizer.bin       # Llama2 词表（已随仓库提供）
│   └── tok512.bin          # 260K 极小词表（可选）
├── src/
│   ├── main.cpp            # generate / chat / HTTP server host 入口
│   ├── decode.cpp / .hpp   # HLS top + host 推理逻辑
│   └── …
├── docs/                   # GitHub Pages 单页前端
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── .nojekyll
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

Linear层采用与 `references/llama2.c/runq.c` 对齐的 Q8_0 W8A8 语义：权重按 `group_size=32` 存储为 INT8 + FP32 scale，activation在linear边界动态量化，32-lane INT8 dot使用INT32累加，再按group scale恢复为FP32输出。Attention RMSNorm后的量化结果由Q/K/V共享，FFN RMSNorm后的量化结果由W1/W3共享。

所有linear顺序复用同一个非模板化`q8_gemv_engine()`。物理datapath一次处理
16个输出行，只从片上parameter tile读取weight/scale；engine按16-row block
遍历当前tile，并用两组accumulator context交替接收下一个block和归约上一个
block。它不知道当前矩阵属于Q/K/V/O、W1/W3/W2还是LM head。外层
`q8_linear()` controller按64个输出行遍历矩阵：先将第一个tile装入URAM，
随后由唯一的非inline `q8_gemv_tile()`在同一个dataflow step中用连续128-bit
burst装载下一个tile，同时顺序计算当前tile。M5.1在当前矩阵最后一个tile计算时
直接预取下一矩阵的第一个tile，Q/K/V/O、W1/W3/W2、下一层Q和LM head之间不再
重复执行独立的首次填充。唯一一组ping/pong URAM在`decode`顶层显式实例化。

RMSNorm、RoPE、attention、residual和KV cache继续使用FP32。CPU reference和FPGA
都按`head_dim = kDim / kNumHeads`执行attention，RoPE只遍历`head_dim / 2`
个pair；每个token只从DDR加载一次该position的24组sin/cos，后续layer复用
RoPE内部buffer。K/V cache统一为`[layer][position][dim]`，每次按128 bit连续
读写完整position。Attention按position依次读取K和V，用online softmax同步更新
每个head的running max、归一化和weighted value，不再物化QK/softmax中间数组，
也不再对V cache执行跨position的窄步长读取。

LM head 不走通用 writeback 路径，专门在 `lm_head` 中流式扫描 embedding rows 并维护 argmax：

```text
final_q = quantize(final_norm)
for tile in embedding_rows:
  scores[64] = q8_gemv_tile(tile, final_q)
  best = max(best, scores)
```

`final_norm[288]` 在LM head开始时只量化一次。LM head仍每获得16个
score就立即更新argmax，不保存32K logits。该实现保留Q8模型的exact
argmax；输入token embedding在host加载checkpoint时反量化为FP32，LM head
则直接扫描Q8 embedding权重。

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

`model/stories15M_q8.bin` 是 `llama2.c` version-2 Q8_0 checkpoint，group size为32；`model/tokenizer.bin`也已包含在仓库中。权重文件头、模型shape、group size和shared classifier标志会在加载时校验。

Host tokenizer的BPE encode/decode语义与`references/llama2.c/runq.c`一致。Prompt
prefill、连续对话状态和HTTP请求都在PS侧管理；PL中的`decode` kernel只接收
`token`和`pos`，不区分host运行模式。

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
syn.interface.m_axi_max_widen_bitwidth=128
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

HTTP server使用与llama.cpp相同的header-only依赖：`cpp-httplib`和
`nlohmann/json`。依赖下载到gitignored的`outputs/deps/`，不进入项目源码：

```bash
mkdir -p outputs/deps
git clone --depth 1 --branch v0.50.1 \
  https://github.com/yhirose/cpp-httplib.git outputs/deps/cpp-httplib
git clone --depth 1 --branch v3.12.0 \
  https://github.com/nlohmann/json.git outputs/deps/json
```

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
  -Ioutputs/deps/cpp-httplib \
  -Ioutputs/deps/json/single_include \
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
    ├── stories15M_q8.bin
    └── tokenizer.bin
```

```bash
PLATFORM_ROOT=/opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1
mkdir -p outputs/bundle/model

cp -f outputs/host/llama2_host outputs/bundle/llama2_host
cp -f outputs/link/binary_container_1.xclbin outputs/bundle/binary_container_1.bin
cp -f "$PLATFORM_ROOT/sw/boot/pl.dtbo" outputs/bundle/pl.dtbo
cp -f model/stories15M_q8.bin outputs/bundle/model/stories15M_q8.bin
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
scp outputs/bundle/model/stories15M_q8.bin \
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

`generate`是默认模式，也可以显式提供prompt：

```bash
./llama2_host -m generate --prompt "Once upon a time" --max_seq 64
```

进入连续对话模式：

```bash
./llama2_host -m chat
```

Chat使用`references/llama2.c/runq.c`的Llama 2 chat模板并连续保留KV cache，
直到达到模型的256-token context。TinyStories checkpoint不是chat模型，因此该模式
只用于验证tokenizer、prompt prefill和连续KV流程，不代表对话质量。

启动本地HTTP服务：

```bash
./llama2_host -m server --host 127.0.0.1 --port 8080
```

Server初始化模型、XRT和FPGA buffer一次，只处理一个请求。当前只提供三个接口：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinystories-15m-w8a8-kv260",
    "prompt": "Once upon a time",
    "max_tokens": 64,
    "temperature": 0,
    "stream": false
  }'
```

`GET /health`在服务可用时返回HTTP 200和`{"status":"ok"}`。
`POST /v1/completions`返回llama.cpp/OpenAI-compatible的
`text_completion` JSON；当前不支持streaming或非零temperature。省略
`max_tokens`时默认上限为64；遇到BOS或EOS会提前结束。

## GitHub Pages

`docs/`是无构建步骤的静态单页前端。页面只显示KV260状态、模型输出和
prompt输入框；固定使用`tinystories-15m-w8a8-kv260`、64-token上限、
greedy argmax和非流式completion。

在KV260运行server后，用ngrok将本地8080端口公开为HTTPS：

```bash
ngrok http 8080
```

当前前端使用KV260 ngrok地址：

```html
<meta name="llama2-api-base"
      content="https://unsacrilegious-abstractive-werner.ngrok-free.dev">
```

最后在GitHub仓库的Settings -> Pages中选择`Deploy from a branch`，发布
`Q8`分支的`/docs`目录。页面地址为：

```text
https://ruhai-lin.github.io/llama2.hls/
```

页面加载时只请求`GET /health`。Board、server或ngrok未运行时，页面显示
`KV260 is resting`并禁用输入。

下表对比M2、M3、M5和当前M5.1，均为KV260单HP0、150 MHz实测。M5和M5.1
每个长度运行3次并报告中位数；M3的1-token结果包含明显启动抖动。`temp=0`时，
16-token输出为`Once upon a time, there was a little girl named Lily. She loved`，
与CPU实现和`references/llama2.c/runq.c`一致。

| Tokens | M2 speed | M3 speed | M5 speed | M5.1 median time | M5.1 speed | M5 -> M5.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.6195 tok/s | 54.0920 tok/s | 78.2472 tok/s | 0.011243 s | 88.9455 tok/s | +13.7% |
| 16 | 73.9004 tok/s | 73.8540 tok/s | 93.5924 tok/s | 0.141770 s | 112.859 tok/s | +20.6% |
| 64 | 42.1067 tok/s | 42.0031 tok/s | 80.4145 tok/s | 0.583109 s | 109.756 tok/s | +36.5% |

测完后可恢复默认 starter app：

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## PPA 与瓶颈

本节的资源、时序与层级数据来自当前M5.1源码的Vitis HLS 2025.2综合和
150 MHz routed implementation。

### 资源与时序

| Scope | LUT | FF/REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| M5.1 single-engine HLS estimate | 62,297 (53%) | 64,508 (27%) | 72 BRAM18K (25%) | 16 (25%) | 148 (11%) |
| M5.1 linked kernel synthesis | 55,532 (47.41%) | 57,394 (24.50%) | 25.5 tiles (17.71%) | 16 (25%) | 162 (12.98%) |

HLS对quantize中的局部组合路径给出6.727 ns的保守估算；150 MHz完整物理实现
的实际routed WNS为`+0.341 ns`，WHS为`+0.010 ns`，所有用户时序约束满足。

### HLS Cycle Bottleneck

| Module | HLS latency |
|---|---:|
| Shared `q8_gemv_tile` | 4,172 cycles/call |
| Dataflow `q8_gemv_step` | 4,171 cycles/call |
| Parameter tile load | 4,171 cycles/call |
| Current tile compute | 1,241 cycles/call |
| Shared `q8_gemv_engine` | 1,240 cycles/call |
| quantize 288 | 937 cycles |
| quantize 768 | 2,497 cycles |

HLS hierarchy和routed netlist中都只有一个`q8_gemv_tile`、一个
`q8_gemv_step`、一个loader、一个compute controller和一个
`q8_gemv_engine`。该tile内只有一套2x32-lane INT8 MAC、两组交替使用的
accumulator context和一套reduction/dequant/FP32 accumulation datapath。
routed timing hierarchy也只出现一个engine实例。

片上parameter storage固定为`ping/pong x 4 banks x 864 words x 128 bit`。HLS推导出
8个独立URAM bank，每个bank深度864、宽度128 bit并占用2个URAM，总计16个URAM。
controller先填充ping；稳态下loader写入空闲tile的同时，唯一的engine从另一个tile
计算最多64个输出行，然后交换ping/pong。HLS dataflow报告中full-tile load
为4,171 cycles、compute为1,241 cycles，step仍为4,171 cycles，证明
两者已重叠且当前稳态关键路径是参数装载。M5.1进一步用当前矩阵最后一次compute
覆盖下一矩阵的首次tile load，消除了43次linear调用之间的42个参数断层。

当前实现是功能正确的单engine W8A8 M5.1 checkpoint，第一瓶颈仍是LM head。
Q/K/V共享一次activation
quantization，W1/W3也共享一次；各矩阵的weight scale本身不同，不能跨矩阵共享。
checkpoint仍保持llama2.c的split Q8格式。host加载后将全部Q8 weight和scale一次
pack成17,086,464-byte FPGA参数blob；每16 rows按group-major排列，一个group由
1个512-bit scale word和8个512-bit双row weight word组成，不包含padding。host的
canonical blob布局不变，kernel端将每个512-bit word拆成4个连续128-bit beat。
64-row tile对应324个512-bit word/1,296个128-bit beat（`group_count=9`），或864个
512-bit word/3,456个128-bit beat（`group_count=24`）。装载后再执行INT8 dot和8-stage
浮点归约。

### 带宽

HLS interface 报告显示：

| Port group | Width | Burst |
|---|---:|---|
| `m_axi_gmem` | 128 bit | max read burst 256, read outstanding 16 |

kernel只有一个AXI master；Link cfgen将`m_axi_gmem`映射到KV260 base platform的
`S_AXI_HP0_FPD` path。HLS `kernel.xml`和link后的xclbin metadata都显示唯一
`M_AXI_GMEM` data width为128 bit，与HP0物理数据宽度一致。以150 MHz计算，
理想接口上限是`128/8*150MHz = 2.4 GB/s`，实际还受DDR控制器、burst质量和
kernel launch开销影响。

每个group包含32个INT8 weight和一个FP32 scale。packed parameter loader的连续
访问已推断为可变长128-bit burst；`m_axi_max_widen_bitwidth=128`防止HLS将该
单流接口再自动拓宽。生成RTL的AXI adapter固定输出INCR burst，并将长tile
请求拆成最多256 beats。M5将全局read outstanding从2提高到16，AXI adapter
的HLS资源估算由8增至30 BRAM18K；64-row targeted RTL测试从M3的11,110
cycles降至9,334 cycles，减少16.0%。

按每个token扫描17,086,464-byte packed参数计算，M5.1端到端实测对应的有效参数
带宽分别为1-token `1.520 GB/s`、16-token `1.928 GB/s`、64-token
`1.875 GB/s`。position-major online attention消除了原先64-token时明显的
sequence-length退化；16和64-token均越过100 tok/s阶段目标，并已接近单HP0在
该访问模式下的实际带宽上限。

## 参考资料

- [AMD KV260 platform 教程](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-Platform-Creation/Custom-Platform-Creation-Tutorial-on-MPSoC)
- [turingmotors/swan](https://zenn.dev/turing_motors/articles/82505880d27d65)
- [本系列 Vitis 2025 部署笔记](https://blog.csdn.net/Bird_Boss/article/details/151792370)
