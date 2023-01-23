#!/bin/bash
nnUNet_plan_and_preprocess \
-t 608 \
--planner3d ExperimentPlannerTransUNet3D \
--planner2d ExperimentPlannerTransUNet
