# fpga/tcl/build.tcl
# Vivado non-GUI build script.
# Usage: vivado -mode batch -source build.tcl -tclargs <TOP> <BOARD> <PART>
#   TOP   : top module name (e.g. blink_top_nexys)
#   BOARD : nexys | zcu104
#   PART  : Vivado part number (e.g. xc7a100tcsg324-1)
#
# Runs synthesis -> implementation -> bitstream and writes
# utilization and timing reports to fpga/reports/<config>/.

set TOP  [lindex $argv 0]
set BOARD [lindex $argv 1]
set PART [lindex $argv 2]

set PROJ ${TOP}_${BOARD}
set OUT  [file normalize ../reports/${PROJ}]
file mkdir $OUT

create_project $PROJ -part $PART -force

# ------------------------------------------------------------------
# Add RTL sources
# ------------------------------------------------------------------
add_files [glob ../../../rtl/core/*.sv]
add_files [glob ../../../rtl/common/*.sv]
add_files [glob ../../../rtl/cim/*.sv]
add_files [glob ../../../rtl/board/${BOARD}/*.sv]
add_files [glob ../../../rtl/board/${BOARD}/*.v]

# Board constraints
if {[file exists ../${BOARD}/pins.xdc]} {
    add_files -fileset constrs_1 ../${BOARD}/pins.xdc
}

set_property top $TOP [current_fileset]

# ------------------------------------------------------------------
# Synthesis
# ------------------------------------------------------------------
synth_design -name $TOP
report_utilization -hierarchical -file ${OUT}/utilization_hier.rpt
report_timing_summary -file ${OUT}/timing_synth.rpt

# ------------------------------------------------------------------
# Implementation
# ------------------------------------------------------------------
opt_design
place_design
route_design
report_utilization -hierarchical -file ${OUT}/utilization_impl.rpt
report_timing_summary -file ${OUT}/timing_impl.rpt

# ------------------------------------------------------------------
# Bitstream
# ------------------------------------------------------------------
write_bitstream -force ${OUT}/${PROJ}.bit

puts "INFO: Build complete. Reports in $OUT"
