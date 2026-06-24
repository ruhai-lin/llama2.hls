# llama2.hls

Karpathy [tinyllama stories15M](https://huggingface.co/karpathy/tinyllamas) 在 AMD KV260（`xck26-sfvc784-2LV-c`）上的 Vitis HLS 部署。

工具链与板卡环境以 **Vitis / Vivado 2025.2**、官方 **KV260 base platform** 为准。

## 仓库结构

```text
.
├── README.md
├── CMakeLists.txt          # CPU-only 本地构建
├── model/
│   ├── stories15M.bin      # 权重
│   ├── tokenizer.bin       # LLaMA2 tokenizer
│   └── tok512.bin          # 可选小词表
├── src/
│   ├── main.cpp            # host 入口
│   ├── decode.cpp / .hpp   # host 侧 decode 调度
│   ├── tensor_fpga.cpp     # host 侧 FPGA wrapper
│   ├── kernel_*.cpp        # 多个 HLS top
│   └── ...
└── outputs/
    ├── hls/                  # kernel_*.xo
    ├── link/                 # binary_container_1.xclbin
    ├── host/                 # llama2_host
    └── bundle/               # 拷到 KV260 的运行包
```

## 前置环境

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
platforminfo /opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm
```

应看到 `FPGA Device: xck26`、`Board Part: xck26-sfvc784-2LV-c`。

## 模型与词表

`model/tokenizer.bin` 已在仓库中。权重需位于：

```text
model/stories15M.bin
```

如需重新下载：

```bash
cd model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
cd ..
```

## HLS 综合

每个 `kernel_*.cpp` 单独综合为一个 `.xo`：

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
mkdir -p outputs/hls outputs/logs

PLATFORM=/opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm

v++ -c -t hw --platform "$PLATFORM" -k kernel_add     src/kernel_add.cpp     -o outputs/hls/kernel_add.xo
v++ -c -t hw --platform "$PLATFORM" -k kernel_mul     src/kernel_mul.cpp     -o outputs/hls/kernel_mul.xo
v++ -c -t hw --platform "$PLATFORM" -k kernel_rmsnorm src/kernel_rmsnorm.cpp -o outputs/hls/kernel_rmsnorm.xo
v++ -c -t hw --platform "$PLATFORM" -k kernel_softmax src/kernel_softmax.cpp -o outputs/hls/kernel_softmax.xo
v++ -c -t hw --platform "$PLATFORM" -k kernel_rope    src/kernel_rope.cpp    -o outputs/hls/kernel_rope.xo
v++ -c -t hw --platform "$PLATFORM" -k kernel_matmul  src/kernel_matmul.cpp  -o outputs/hls/kernel_matmul.xo
```

预期输出：

```text
outputs/hls/kernel_add.xo
outputs/hls/kernel_mul.xo
outputs/hls/kernel_rmsnorm.xo
outputs/hls/kernel_softmax.xo
outputs/hls/kernel_rope.xo
outputs/hls/kernel_matmul.xo
```

## v++ Link

将 6 个 `.xo` 链接为 KV260 xclbin：

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
mkdir -p outputs/link outputs/logs/link

PLATFORM=/opt/xilinx/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm

v++ -l -t hw \
  --platform "$PLATFORM" \
  outputs/hls/kernel_add.xo \
  outputs/hls/kernel_mul.xo \
  outputs/hls/kernel_rmsnorm.xo \
  outputs/hls/kernel_softmax.xo \
  outputs/hls/kernel_rope.xo \
  outputs/hls/kernel_matmul.xo \
  -o outputs/link/binary_container_1.xclbin \
  --clock.default_freqhz 50000000 \
  --save-temps \
  --temp_dir outputs/link/_x \
  --log_dir outputs/logs/link \
  --report_dir outputs/link/reports
```

预期输出：

```text
outputs/link/binary_container_1.xclbin
```

## Host 交叉编译

```bash
COMMON=/home/ruhai/Projects/KV260_Study/xilinx-zynqmp-common-v2022.2

source /opt/xilinx/2025.2/Vitis/settings64.sh
unset LD_LIBRARY_PATH
source "$COMMON/environment-setup-cortexa72-cortexa53-xilinx-linux"
SYSROOT="$COMMON/sysroots/cortexa72-cortexa53-xilinx-linux"
mkdir -p outputs/host

aarch64-xilinx-linux-g++ -Wall -Wextra -std=c++2a \
  -mcmodel=large -fno-PIC -fno-PIE -no-pie -g --sysroot="$SYSROOT" \
  -Isrc \
  src/context.cpp src/decode.cpp src/main.cpp \
  src/tensor.cpp src/tensor_fpga.cpp src/vocab.cpp src/weight.cpp \
  -L"$SYSROOT/usr/lib" \
  -lxilinxopencl -lxrt_coreutil -lpthread -lrt -ldl \
  -o outputs/host/llama2_host
```

预期输出：

```text
outputs/host/llama2_host
```

## 打包 Bundle

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

预期目录：

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

## 部署到 KV260

传输 bundle：

```bash
KV260=ubuntu@192.168.137.123
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

加载 app：

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

```bash
cd ~/Projects/llama2_bundle
./llama2_host --max_seq 16 --temp 0
```

KV260 实测：

| Command | Host-reported time | Host-reported speed | Wall time |
|---|---:|---:|---:|
| `./llama2_host --max_seq 16 --temp 0` | 16.2322 s | 0.9857 tok/s | 39.91 s |

测试结束后可恢复默认 starter app：

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## 备注

- 这个版本的目标是验证 llama2 风格多 kernel 流程能在 KV260 上跑通，不追求性能。
- `decode.cpp` 仍在 host 侧执行 decode 控制流，LM head logits 和 argmax 也在 host 侧完成。
- `Time`/`Speed` 是程序内部用 `clock()` 统计的 decode 段 CPU 时间；`/usr/bin/time` 的 wall time 包含 xclbin load、OpenCL setup、模型文件读入和程序退出。

## 参考资料

- [AMD KV260 platform tutorial](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-Platform-Creation/Custom-Platform-Creation-Tutorial-on-MPSoC)
- [turingmotors/swan](https://zenn.dev/turing_motors/articles/82505880d27d65)
- [本系列 Vitis 2025 部署笔记](https://blog.csdn.net/Bird_Boss/article/details/151792370)