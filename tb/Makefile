# tb/Makefile - cocotb test runner
SIM      ?= icarus
TOPLEVEL ?=
MODULE   ?=
VERILOG_SOURCES = $(wildcard ../rtl/core/*.sv) $(wildcard ../rtl/common/*.sv) $(wildcard ../rtl/cim/*.sv) $(wildcard ../rtl/board/nexys/*.sv)

include $(shell cocotb-config --makefiles)/Makefile.sim
