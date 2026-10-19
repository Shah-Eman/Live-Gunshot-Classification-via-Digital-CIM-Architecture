# CIM for Deep Learning: FYP Repository

This repository holds a compute-in-memory (CIM) macro emulator for deep learning on FPGA. It is a Final Year Project at UET Lahore, supervised by Prof. Dr. Muhammad Tahir.

The project builds a bit-serial, weight-stationary CIM tile in hand-written RTL and maps a real gunshot-classification 1D-CNN onto it. It then compares the result against a conventional (von Neumann) baseline engine, across different MAC counts, on two FPGA boards.

---

## Repository layout

```
.
├── Makefile                  Top-level build targets: sim, synth, bit
├── requirements.txt          Python dependencies
├── .gitignore
│
├── sw/                       Python software
│   (models, integer reference, test-vector generator,
│    frontend detector, cycle model, quantization)
│
├── tb/                       Cocotb testbenches
│   └── Makefile              Cocotb test runner
│
├── hls/                      Vitis HLS sources (plain C, stdint.h types)
│
├── host/                     Host-side scripts
│   (run_tile.py, run_layer1.py, demo_gui.py)
│
├── results/
│   └── runs.csv              Every measured result, one row per config
│
├── docs/
│   ├── rules.md              Team rules
│   ├── arith_spec.md         Exact integer arithmetic of the whole network
│   ├── decisions.md          Design decisions, with options and reasons
│   ├── diagrams/             .drawio sources and exported .svg
│   (still to come: mapping_spec, baseline_spec, measurement_protocol,
│    file_formats, uart_protocol, resources, dataset_audit)
│
├── rtl/
│   ├── core/                 Board-agnostic design
│   ├── cim/                  CIM tile and requantization
│   │   (cim_tile.sv, requant.sv)
│   ├── common/               Shared blocks
│   │   (adder_tree.sv)
│   └── board/
│       ├── nexys/            Nexys A7-100T wrapper (clock, pins, UART)
│       └── zcu104/            ZCU104 wrapper
│
└── fpga/
    ├── tcl/
    │   └── build.tcl         Non-GUI Vivado build script
    ├── nexys/                Nexys A7 constraints and TCL
    ├── zcu104/               ZCU104 constraints and TCL
    └── reports/              Synthesis & timing reports per config
```

### Design principle: board-agnostic core

`rtl/core/` contains nothing board-specific. Board wrappers in
`rtl/board/nexys/` and `rtl/board/zcu104/` are thin: clocks, pins, and UART
or AXI. The same core runs on both boards.

---

## Two boards

| | Nexys A7-100T | ZCU104 |
|---|---|---|
| **Role** | Development platform | Final platform |
| **Vivado part** | `xc7a100tcsg324-1` | `xczu7ev-ffvc1156-2-e` |
| **Family** | Artix-7 (FPGA only) | Zynq UltraScale+ MPSoC |
| **LUTs** | 63,400 | 230,400 |
| **Block RAM** | 4,860 Kb | 312 × 36 Kb + 96 UltraRAM |
| **DSP slices** | 240 | 1,728 |

Design to the smaller board first. If the model and engine fit the Nexys,
they fit the ZCU104. Never mix numbers from the two boards in one comparison.

---

## Prerequisites

### Python

```bash
pip install -r requirements.txt
```

Installs: numpy, tensorflow, scikit-learn, librosa, soundfile, matplotlib,
pandas, cocotb, pyserial.

### Simulator

Install **Icarus Verilog** and **GTKWave**:

```bash
# Debian/Ubuntu
sudo apt install iverilog gtkwave

# macOS
brew install icarus-verilog gtkwave
```

(Verilator is an acceptable alternative. Use Xcelium later if Cadence is
confirmed.)

### Vivado

Install **Vivado ML Standard** (free) with device support for both
**Artix-7** and **Zynq UltraScale+ MPSoC**. Confirm the ZU7EV part is
available under your licence.

---

## Build system

All builds run from the command line. No GUI.

### Simulation

```bash
make sim
```

Runs the cocotb testbench via Icarus Verilog. Override the simulator:

```bash
make sim SIMULATOR=verilator
```

### Synthesis

```bash
make synth TOP=<top_module> BOARD=nexys
```

Runs synthesis only. Reports are written to `fpga/reports/<TOP>_<BOARD>/`.

### Bitstream

```bash
make bit TOP=<top_module> BOARD=nexys
```

Runs the full flow: synthesis → implementation → bitstream. The `.bit` file
and utilization/timing reports appear in `fpga/reports/<TOP>_<BOARD>/`.

For the ZCU104:

```bash
make bit TOP=<top_module> BOARD=zcu104
```

### Clean

```bash
make clean
```

---

## runs.csv

`results/runs.csv` is the single source of truth for every measured number.
Header:

```
date,commit,board,stage,config,n_macs,f_clk_mhz,cycles,ii,lut,ff,bram,dsp,dsp_array,acc,bitexact_pct,notes
```

| Column | Meaning |
|---|---|
| `date` | Date of the run |
| `commit` | Git commit hash |
| `board` | `nexys` or `zcu104` |
| `stage` | Pipeline stage (e.g. `1.10`) |
| `config` | Configuration name |
| `n_macs` | Number of parallel MAC units |
| `f_clk_mhz` | Clock frequency in MHz |
| `cycles` | Measured cycle count |
| `ii` | Initiation interval |
| `lut` | LUT utilization |
| `ff` | Flip-flop utilization |
| `bram` | Block RAM utilization |
| `dsp` | Whole-design DSP total |
| `dsp_array` | DSP count inside the CIM tile array (must be 0 for CIM) |
| `acc` | Accuracy (%) |
| `bitexact_pct` | Percentage of bit-exact outputs |
| `notes` | Free-text notes |

---

## Team

| Member | Role |
|---|---|
| Shah Eman | Team leader |
| Abubakar | Member |
| Abdul Nafeh | Member |
| Talha Omar | Member |
| Prof. Dr. M. Tahir | Supervisor |

---

## Team rules

1. Every measured number goes into `results/runs.csv` with its commit hash.
2. Nothing in the report is typed by hand.
3. No LLM ever writes a number into the results.

See `docs/rules.md`.

---

## Project stages

| Stage | Description | Weeks |
|---|---|---|
| **0** | Setup and resources | 1–2 |
| **1** | Real engine, first layer, tested part by part | 1–13 |
| **2** | Real data and real model | 3–11 |
| **3** | Scale the engine and build the baseline | 8–15 |
| **4** | FPGA implementation and board | 12–17 |
| **5** | HLS and Cadence | 3–19 |
| **6** | Measurement, then claims | 13–17 |
| **7** | Demo polish, report, defence | 6–20 |

**Critical path:**
0.3 → 1.1 → 1.2 → 1.3 → 1.6 → 1.8 → 1.9 → 1.10 → 3.1 → 3.4 → 3.5 → 4.1 → 4.2 → 6.2 → 6.4 → 7.x

**Milestone A** (Week 8): First real layer works on the board.
**Milestone B** (Week 15): Full model works on hardware.

---

## The CIM tile

The core of the project is `rtl/cim/cim_tile.sv` (Step 1.6):

- **16 × 16** tile (R=16 rows, C=16 columns)
- **Bit-serial**: 8 cycles per dot product, one bit-plane per cycle
- **Weight-stationary**: weights preloaded into distributed RAM
- **No DSP slices** inside the tile array (`(* use_dsp = "no" *)`)
- Per cycle: AND mask → column adder tree → shift-accumulate with MSB subtract
- Requantization lives outside the tile in `rtl/cim/requant.sv` and may use DSP

---

## Key documents

| Document | What it is for |
|---|---|
| `docs/arith_spec.md` | The single definition of every integer operation in the network. Python, C and RTL must all match it bit for bit. |
| `docs/decisions.md` | Each design decision written down with its options and reason, so nobody has to argue it again later. |
| `docs/rules.md` | The three team rules. |

---

## Network

A small 1D-CNN on the raw waveform (defined in `sw/models/cnn1d.py`):

```
Input 8000×1 → Conv1D k64 s8 16ch → MaxPool4
→ Conv1D k9 32ch → MaxPool4
→ Conv1D k9 32ch → GlobalAvgPool → Dense 32→5
```

Layers are referred to by name in the docs and code, not by number: `conv1`, `pool1`, `conv2`, `pool2`, `conv3`,
`gap`, `dense` and `argmax`. The runbook counts only the conv layers, so its "layer 1, 2, 3" are `conv1`, `conv2` and
`conv3`. Using names avoids mixing the two counting schemes up. The full layer table is in `docs/arith_spec.md` §5.1.

- About 2.7 M MACs (2,731,168 exactly) and about 15 k parameters (15,008 weights).
- Tile-friendly shapes: output channels and k×Cin are multiples of 16, so no part of the 16 × 16 tile sits idle.
- Budget: at most 4 M MACs and at most 30 k parameters, so the model fits the LUTRAM of the Nexys A7-100T.
- Quantized to 8-bit integers (symmetric, with per-channel weight scales and per-tensor activation scales).

---

## License

This is a university FYP repository. All rights reserved.
