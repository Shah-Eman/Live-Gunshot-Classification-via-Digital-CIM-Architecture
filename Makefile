# Makefile for CIM FYP project
# Targets: sim, synth, bit
# Usage:   make sim
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

RTL_CORE := $(wildcard rtl/core/*.sv) $(wildcard rtl/common/*.sv) $(wildcard rtl/cim/*.sv)
RTL_BOARD := $(wildcard rtl/board/$(BOARD)/*.sv)
RTL := $(RTL_CORE) $(RTL_BOARD)

.PHONY: sim synth bit clean

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
