from copy import deepcopy

import torch
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
from nnunet.network_architecture.generic_transUNet import GenericTransUNet
from nnunet.network_architecture.transunet.vit_seg_configs import get_r50_b16_config
from nnunet.paths import *
import numpy as np


class ExperimentPlannerTransUNet(ExperimentPlanner2D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerTransUNet, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_transUNet"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlans_transUNet_plans_2D.pkl")


    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 config):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        t = torch.cuda.get_device_properties(0).total_memory
        batch_size = self.unet_min_batch_size
        while GenericTransUNet.use_this_for_batch_size_computation_2D * (batch_size + 2) < t:
            batch_size += 2
        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This should not happen")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        config.img_size = input_patch_size

        plan = {
            "config": config,
            'batch_size': batch_size,
            'patch_size': input_patch_size,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': False
        }
        return plan


    def plan_experiment(self):
        print("!! Hard-coded !! Use R50-ViT-B16 config")
        
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        config = get_r50_b16_config(len(all_classes) + 1)

        target_spacing = self.get_target_spacing()
        new_shapes = np.array([np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)])

        # max_spacing_axis = np.argmax(target_spacing)
        # remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        # self.transpose_forward = [max_spacing_axis] + remaining_axes
        # self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]
        self.transpose_forward = [0, 1, 2] # TODO: hard-coded values ?
        self.transpose_backward = [0, 1, 2] # TODO: hard-coded values ?

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = []

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        self.plans_per_stage.append(
            self.get_properties_for_stage(target_spacing_transposed, target_spacing_transposed, median_shape_transposed,
                                          num_cases=len(self.list_of_cropped_npz_files),
                                          config=config),
            )

        print(self.plans_per_stage)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        normalization_schemes = self.determine_normalization_scheme()
        # deprecated
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        # Determine number of batches
        split_ratio = 0.8
        num_train_sample = int(\
            len(self.list_of_cropped_npz_files) * split_ratio * median_shape_transposed[0])
        num_train_batches_per_epoch = int(num_train_sample / self.plans_per_stage[0]["batch_size"])
        num_val_sample = int(\
            len(self.list_of_cropped_npz_files) * median_shape_transposed[0]) - num_train_sample
        num_val_batches_per_epoch = int(num_val_sample / self.plans_per_stage[0]["batch_size"])

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 "num_train_batches_per_epoch": num_train_batches_per_epoch,
                 "num_val_batches_per_epoch": num_val_batches_per_epoch
                 }

        self.plans = plans
        self.save_my_plans()
