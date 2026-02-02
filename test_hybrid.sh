#!/bin/bash
#$ -N test_hybrid_physics
#$ -q gpu
#$ -l h=node3b07
cd ~/mumax3_saf_build_01022026/mumax3-neurospin/tests/core
module load cuda/11.8
../../../mumax3-saf-neurospin-v2.1.0 test_simple_2layer.mx3
echo === Physics validation ===
grep mx0.*mx1 simple_final.log
