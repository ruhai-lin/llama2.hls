# llama2.hls

Karpathy [tinyllama stories15M](https://huggingface.co/karpathy/tinyllamas) 在 AMD KV260（`xck26-sfvc784-2LV-c`）上的 Vitis HLS 部署。Host 在 ARM 上跑采样与打印；FPGA 侧为单个 `kernel_decode` megakernel，每个 token 只启动一次设备 kernel，权重与 KV cache 常驻 DDR，kernel 内完成 decode forward 与 LM head。

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
│   ├── kernel_*.cpp        # 被 decode.cpp include 的 megakernel 内部模块
│   └── …
└── outputs/                # 本地构建产物（git ignore）
    ├── hls/ link/ host/ bundle/ logs/
```

`kernel_*.cpp` 文件名与 [swan](https://zenn.dev/turing_motors/articles/82505880d27d65) 参考实现对齐，由 `decode.cpp` 直接 include 进同一 HLS 翻译单元，不分别生成多个 `.xo`。

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

将 `<kv260-ip>` 换成板卡地址（示例 `192.168.137.208`）。

```bash
KV260=ubuntu@<kv260-ip>
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

```text
Time : 5.06948[s]
Speed: 3.15615[tok/s]
```

测完后可恢复默认 starter app：

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## 参考资料

- [AMD KV260 platform 教程](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-Platform-Creation/Custom-Platform-Creation-Tutorial-on-MPSoC)
- [turingmotors/swan](https://zenn.dev/turing_motors/articles/82505880d27d65)
- [本系列 Vitis 2025 部署笔记](https://blog.csdn.net/Bird_Boss/article/details/151792370)
