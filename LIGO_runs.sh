#!/bin/bash

#Define the base directory
BASE_DIR="/data/hlssynt-users/ziyin/vivado_runs/LIGO/"

cd $BASE_DIR

for PREFIX in r1 r2 r4; do

	for PROJECT_DIR in ${PREFIX}_10int_{5..10}frac; do
        	echo "Processing $PROJECT_DIR"

        	#Change to the project directory
        	cd $PROJECT_DIR

        	vivado_hls -f build_prj.tcl "reset=0 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0 fifo_opt=0"

        	vivado_hls -f build_prj.tcl "reset=0 csim=0 synth=0 cosim=0 validation=0 export=0 vsynth=1 fifo_opt=0"

        	rm -rf myproject_prj tb_data

        	cd $BASE_DIR
	done
done

echo "All projects processed."

