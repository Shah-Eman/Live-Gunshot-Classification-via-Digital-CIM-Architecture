# Makefile for CIM FYP project
# Targets: test_py, tile_demo, model_table, train_quick, sim, synth, bit, clean
# Usage:   make test_py                  (Python tests, no tools needed beyond numpy)
#          make tile_demo                (print the spec section 7 worked example)
#          make model_table              (per-layer table of cnn1d.py, checked against spec 5.1)
#          make train_quick MANIFEST=... EDGE_LABELS=...   (Step 1.1 quick model)
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

.PHONY: test_py tile_demo model_table train_quick sim synth bit clean

# ------------------------------------------------------------------
# Python tests (Step 1.4 tile model; later steps add their tests here)
# ------------------------------------------------------------------
test_py:
	$(PYTHON) -m unittest discover -s sw/ref -p 'test_*.py' -v

# Print the arith_spec.md section 7 worked example, cycle by cycle
tile_demo:
	$(PYTHON) sw/ref/tile_model.py

# ------------------------------------------------------------------
# The network (Step 1.1)
# ------------------------------------------------------------------
# Fails if the table printed by cnn1d.py disagrees with arith_spec.md 5.1
model_table:
	$(PYTHON) sw/models/cnn1d.py

MANIFEST    ?= ../Dataset-Final/manifest.csv
EDGE_LABELS ?= ../Dataset/edge-collected-gunshot-audio/gunshot-audio-labels-only.csv

# Quick sanity-check model; never quote its accuracy
train_quick:
	$(PYTHON) sw/models/train_quick.py --manifest "$(MANIFEST)" --edge-labels "$(EDGE_LABELS)"

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
