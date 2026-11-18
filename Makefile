# Makefile for CIM FYP project
# Targets: test_py, tile_demo, model_table, train_quick, step16_checks, sim, synth, bit, clean
# Usage:   make test_py                  (Python tests, no tools needed beyond numpy)
#          make tile_demo                (print the spec section 7 worked example)
#          make model_table              (per-layer table of cnn1d.py, checked against spec 5.1)
#          make train_quick              (Step 1.1 quick model)
#          make step16_checks            (Step 1.6 tests, split check, window check; needs the dataset)
#          make sim
#          make synth TOP=<top_module> BOARD=nexys
#          make bit   TOP=<top_module> BOARD=nexys

BOARD ?= nexys
TOP   ?=

# Board to Vivado part number mapping
ifeq ($(BOARD),nexys)
PART := xc7a100tcsg324-1
else ifeq ($(BOARD),zcu104)
PART := xczu7ev-ffvc1156-2-e
else
$(error Unknown BOARD '$(BOARD)'. Use BOARD=nexys or BOARD=zcu104)
endif

SIMULATOR ?= icarus
PYTHON    ?= python3

RTL_CORE := $(wildcard rtl/core/*.sv) $(wildcard rtl/common/*.sv) $(wildcard rtl/cim/*.sv)
RTL_BOARD := $(wildcard rtl/board/$(BOARD)/*.sv)
RTL := $(RTL_CORE) $(RTL_BOARD)

.PHONY: test_py tile_demo model_table train_quick sim synth bit clean \
        test_frontend windows scan splits check_splits check_windows listen step16_checks

# ------------------------------------------------------------------
# Python tests (Step 1.4 tile model; later steps add their tests here)
# ------------------------------------------------------------------
test_py:
	$(PYTHON) -m unittest discover -s sw/ref -p 'test_*.py' -v

# Print the arith_spec.md section 7 worked example, cycle by cycle
tile_demo:
	$(PYTHON) sw/ref/tile_model.py

# ------------------------------------------------------------------
# Step 1.6: detector, windows and splits (docs/frontend.md, D22, D23)
# Paths may contain spaces; override on the command line, e.g.
#   make check_windows DATASET="/path/to/Dataset-Final" PYTHON=../.venv-audio/bin/python
# ------------------------------------------------------------------
DATASET          ?= ../Dataset-Final
OUT              ?= ../step16_outputs
EDGE_LABELS      ?= sw/data/meta/gunshot-audio-all-metadata.csv
EDGE_LABELS_ONLY ?= sw/data/meta/gunshot-audio-labels-only.csv
SPLITS           ?= sw/data/splits.json
RESULTS          ?= results/step16
MANIFEST         ?= $(DATASET)/manifest.csv

test_frontend:
	CIM_MANIFEST="$(MANIFEST)" $(PYTHON) -m unittest discover -s sw/frontend -p 'test_*.py' -v

windows:
	$(PYTHON) sw/frontend/cut_windows.py --manifest "$(MANIFEST)" --edge-labels "$(EDGE_LABELS)" \
	  --out "$(OUT)" --report-dir "$(RESULTS)"

scan:
	$(PYTHON) sw/frontend/scan_clips.py --manifest "$(MANIFEST)" --out "$(OUT)" --report-dir "$(RESULTS)"

splits:
	$(PYTHON) sw/data/make_splits.py --manifest "$(MANIFEST)" --out "$(SPLITS)"

check_splits:
	$(PYTHON) sw/data/check_splits.py --manifest "$(MANIFEST)" --splits "$(SPLITS)" \
	  --window-index "$(RESULTS)/window_index.csv" --report "$(RESULTS)/splits_report.md"

check_windows:
	$(PYTHON) sw/frontend/check_windows.py --manifest "$(MANIFEST)" --edge-labels "$(EDGE_LABELS)" \
	  --splits "$(SPLITS)" --window-index "$(RESULTS)/window_index.csv" \
	  --background-index "$(RESULTS)/background_index.csv" --gun-npz "$(OUT)/gun_windows.npz" \
	  --report "$(RESULTS)/check_windows_report.md"

listen:
	$(PYTHON) sw/frontend/export_listen.py --manifest "$(MANIFEST)" --splits "$(SPLITS)" \
	  --window-index "$(RESULTS)/window_index.csv" --background-index "$(RESULTS)/background_index.csv" \
	  --gun-npz "$(OUT)/gun_windows.npz" --out "$(OUT)/listen"

step16_checks: test_frontend check_splits check_windows

# ------------------------------------------------------------------
# The network (Step 1.1)
# ------------------------------------------------------------------
# Fails if the table printed by cnn1d.py disagrees with arith_spec.md 5.1
model_table:
	$(PYTHON) sw/models/cnn1d.py

# Quick sanity-check model; never quote its accuracy
train_quick:
	$(PYTHON) sw/models/train_quick.py --manifest "$(MANIFEST)" --edge-labels "$(EDGE_LABELS_ONLY)"

# ------------------------------------------------------------------
# Simulation (cocotb + Icarus Verilog)
# ------------------------------------------------------------------
sim:
	cd tb && \
	SIM=$(SIMULATOR) TOPLEVEL=$(TOP) MODULE=$(TEST_MODULE) make sim

# ------------------------------------------------------------------
# Synthesis only
# ------------------------------------------------------------------
synth:
	cd fpga/tcl && vivado -mode batch -source build.tcl \
	-tclargs $(TOP) $(BOARD) $(PART) -log vivado.log -journal vivado.jou

# ------------------------------------------------------------------
# Full build to bitstream
# ------------------------------------------------------------------
bit:
	cd fpga/tcl && vivado -mode batch -source build.tcl \
	-tclargs $(TOP) $(BOARD) $(PART) -log vivado.log -journal vivado.jou

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
clean:
	rm -rf fpga/reports/* tb/sim_build/ tb/*.xml tb/*.vcd tb/*.fst
