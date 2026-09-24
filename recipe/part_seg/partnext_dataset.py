"""
PartNeXt 3D Part Segmentation Dataset.

Loads PartNeXt data directly from GLB files + face-level mask annotations.
Parquet files are produced by prepare_partnext_dataset.py and contain:
  - glb_path: path to the GLB file
  - masks_json / mesh_face_num_json: face-level annotation
  - target_mask_ids: per-part list of maskIds (with hierarchy expansion)
  - messages / diffusion_messages: conversation-format training text
"""

import ast
import io
import igl
import json
import logging
import os
import random
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import trimesh

import importlib.util as _imp_util

def _import_from_path(module_name, file_path):
    """Import a module from an absolute file path without modifying sys.path."""
    spec = _imp_util.spec_from_file_location(module_name, file_path)
    mod = _imp_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mesh_utils = _import_from_path("mesh_utils", os.path.join(os.path.dirname(__file__), "mesh_utils.py"))
scene2meshes = _mesh_utils.scene2meshes
normalize_meshes_diag = _mesh_utils.normalize_meshes_diag
reorder_meshes_by_face_num = _mesh_utils.reorder_meshes_by_face_num

_aug_mod = _import_from_path("augmentation", os.path.join(os.path.dirname(__file__), "augmentation.py"))
augment_point_cloud = _aug_mod.augment_point_cloud
transform_bbox_aabb = _aug_mod.transform_bbox_aabb

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


OPEN_TEMPLATES = {
    "coarse": {
        "semantic": [
            "Please roughly segment this 3D object in <point_cloud>.",
            "Please segment the main structural parts of this 3D object in <point_cloud>.",
            "Please identify the major components of this 3D object in <point_cloud>.",
        ],
        "numeric": [
            "Please segment this 3D object into about {n} major parts in <point_cloud>.",
        ],
        "weights": (0.8, 0.2),
    },
    "medium": {
        "semantic": [],
        "numeric": [
            "Please segment this 3D object into about {n} parts in <point_cloud>.",
            "Please segment this 3D object into approximately {n} parts in <point_cloud>.",
        ],
        "weights": (0.0, 1.0),
    },
    "fine": {
        "semantic": [
            "Please segment all the detailed parts of this 3D object in <point_cloud>.",
            "Please finely segment every part of this 3D object in <point_cloud>.",
            "Please segment this 3D object into fine-grained parts in <point_cloud>.",
        ],
        "numeric": [
            "Please segment this 3D object into about {n} detailed parts in <point_cloud>.",
        ],
        "weights": (0.8, 0.2),
    },
}



OPEN_GENERIC_PROB = 0.10
OPEN_GENERIC_TEMPLATES = [
    "Please segment all parts of this 3D object in <point_cloud>.",
    "Please segment this 3D object in <point_cloud>.",
]


SEMANTIC_OPEN_TEMPLATES = {
    "coarse": "Please segment and name the main parts of this 3D object in <point_cloud>.",
    "fine": "Please segment and name all the detailed parts of this 3D object in <point_cloud>.",
    "numeric": "Please segment and name about {n} parts of this 3D object in <point_cloud>.",
    "generic": "Please segment and name all parts of this 3D object in <point_cloud>.",
}
NOSEM_OPEN_TEMPLATES = {
    "coarse": "Please segment this 3D object into its main geometric parts in <point_cloud>.",
    "fine": "Please segment this 3D object into detailed geometric parts in <point_cloud>.",
    "numeric": "Please segment this 3D object into about {n} geometric parts in <point_cloud>.",
    "generic": "Please segment this 3D object into geometric parts in <point_cloud>.",
}
LEGACY_SEMANTIC_TEMPLATES = OPEN_GENERIC_TEMPLATES
LEGACY_NOSEM_TEMPLATES = [
    "Please segment this 3D object into parts in <point_cloud>.",
    "Please divide this 3D object into geometric regions in <point_cloud>.",
]
SEMANTIC_PROMPTABLE_TEMPLATE = "Please segment and name the part at {point} in <point_cloud>."
NOSEM_PROMPTABLE_TEMPLATE = "Please segment the geometric part at {point} in <point_cloud>."


def _classify_nosem_bucket(num_parts: int) -> str:
    """Choose a geometric granularity bucket from the requested part count."""
    if num_parts <= 5:
        return "coarse"
    if num_parts <= 15:
        return "medium"
    return "fine"



def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        return data_item


class PartNeXtPoint3DDataset(Dataset):
    """
    Dataset for PartNeXt 3D part segmentation.

    Loads GLB files directly at training time and builds point cloud + segmentation masks.

    Args:
        parquet_files: Path(s) to parquet files produced by prepare_partnext_dataset.py.
        tokenizer: Tokenizer for text.
        config: DictConfig with training options.
        processor: Multimodal processor (Qwen3VL).
        max_samples: Limit number of samples (-1 = all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        is_train: bool = True,
    ):
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"]
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.messages_key = config.get("messages_key", "messages")
        self.diffusion_messages_key = config.get("diffusion_messages_key", "diffusion_messages")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.is_train = bool(is_train)
        self.aug_config = config.get("augmentation", {})
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        assert self.truncation in ["error", "left", "right"]


        self._hf_cfg = None
        self.point_cloud_encoder_type = config.get("point_cloud_encoder_type", None)
        if self.point_cloud_encoder_type is None and processor is not None:
            try:
                from transformers import AutoConfig
                self._hf_cfg = AutoConfig.from_pretrained(
                    processor.tokenizer.name_or_path, trust_remote_code=True
                )
                self.point_cloud_encoder_type = getattr(self._hf_cfg, "point_cloud_encoder_type", "utonia")
            except Exception:
                self.point_cloud_encoder_type = "utonia"
        elif self.point_cloud_encoder_type is None:
            self.point_cloud_encoder_type = "utonia"
        if self.point_cloud_encoder_type != "utonia":
            raise ValueError("PartLLM supports only the Utonia point-cloud encoder")

        from transformers.models.qwen3_vl import utonia as _utonia
        utonia_scale = config.get("utonia_scale", None)
        if utonia_scale is None and self._hf_cfg is not None:
            utonia_scale = getattr(self._hf_cfg, "utonia_scale", 5.0)
        utonia_scale = utonia_scale or 5.0
        apply_z_positive = bool(config.get("utonia_apply_z_positive", False))
        self.utonia_transform = _utonia.transform.default(
            utonia_scale,
            normalize_coord=True,
            apply_z_positive=apply_z_positive,
        )
        utonia_num_tokens = config.get("utonia_num_tokens", None)
        if utonia_num_tokens is None and self._hf_cfg is not None:
            utonia_num_tokens = getattr(self._hf_cfg, "utonia_num_tokens", None)
        if utonia_num_tokens is None:
            utonia_num_tokens = 25000
            print(f"[Utonia] No utonia_num_tokens configured, using default: {utonia_num_tokens}")
        self.utonia_num_tokens = utonia_num_tokens


        pc_size = config.get("pc_size", None)
        if pc_size is None and self._hf_cfg is not None:
            pc_cfg = getattr(self._hf_cfg, "point_cloud_config", None)
            if pc_cfg is not None:
                pc_size = pc_cfg.get("pc_size", None) if isinstance(pc_cfg, dict) else getattr(pc_cfg, "pc_size", None)
            if pc_size is None:
                pc_size = getattr(self._hf_cfg, "pc_size", 81920)
        self.pc_size = pc_size or 81920

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        if self.processor is not None:
            self.processor.num_pc_tokens = self.utonia_num_tokens

        self._read_files_and_process()

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            if "_dataset_source" not in dataframe.columns:
                lower_path = str(parquet_file).lower()
                source = "3dcompat200" if "compat" in lower_path else "partnext"
                dataframe["_dataset_source"] = source
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"[PartNeXt] dataset len: {total}")

        if self.max_samples is not None and self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"[PartNeXt] selected {self.max_samples} random samples out of {total}")

        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def __len__(self):
        return len(self.messages)





    def _load_from_npz(self, path: str, mesh_face_num_json=None):
        """Load HY3D-Bench part meshes directly from one official NPZ file."""
        data = np.load(path, allow_pickle=True)
        part_keys = sorted(
            [key for key in data.keys() if key.startswith("part_") and key.endswith(".ply")],
            key=lambda key: int(re.search(r"(\d+)", key).group()),
        )
        if not part_keys:
            raise ValueError(f"No part_*.ply entries found in {path}")
        mesh_list = []
        for key in part_keys:
            mesh = trimesh.load(
                io.BytesIO(data[key].tobytes()), file_type="ply", process=False
            )
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError(f"Invalid mesh entry {key} in {path}")
            mesh_list.append(mesh)
        mesh_list = normalize_meshes_diag(mesh_list, norm_diag_len=1.0)
        for mesh in mesh_list:
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.full((len(mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
            )
        combined_mesh = trimesh.util.concatenate(mesh_list)
        sampled_points, face_idx, sampled_colors = trimesh.sample.sample_surface(
            combined_mesh, self.pc_size, seed=None, sample_color=True
        )
        return sampled_points, sampled_colors, face_idx, combined_mesh, mesh_list

    def _load_and_sample_point_cloud(self, glb_path: str, mesh_face_num_json=None):
        """
        Load GLB, normalize, reorder meshes to match annotation, and sample point cloud.

        Args:
            glb_path: path to GLB file
            mesh_face_num_json: annotation's mesh_face_num (str or dict) for reordering

        Returns:
            sampled_points: [pc_size, 3] numpy array
            sampled_colors: [pc_size, 4] numpy uint8 array (RGBA)
            face_idx: [pc_size] numpy array of face indices into combined mesh
            combined_mesh: the concatenated trimesh.Trimesh
            mesh_list: list of individual normalized trimesh.Trimesh (reordered)
        """
        if str(glb_path).lower().endswith(".npz"):
            return self._load_from_npz(glb_path, mesh_face_num_json)

        scene = trimesh.load(glb_path, force="scene")
        mesh_list = scene2meshes(scene)
        if not mesh_list:
            raise ValueError(f"No meshes found in {glb_path}")
        mesh_list = normalize_meshes_diag(mesh_list, norm_diag_len=1.0)


        if mesh_face_num_json is not None:
            mesh_face_num = ast.literal_eval(mesh_face_num_json) if isinstance(mesh_face_num_json, str) else mesh_face_num_json
            mesh_list = reorder_meshes_by_face_num(mesh_list, mesh_face_num)


        for i, m in enumerate(mesh_list):
            if m.visual.kind == "texture":
                try:
                    mesh_list[i].visual = m.visual.to_color()
                except (IndexError, ValueError):

                    nv = len(mesh_list[i].vertices)
                    mesh_list[i].visual = trimesh.visual.ColorVisuals(
                        vertex_colors=np.full((nv, 4), [102, 102, 102, 255], dtype=np.uint8)
                    )

            vc = mesh_list[i].visual.vertex_colors
            nv = len(mesh_list[i].vertices)
            if vc.ndim == 1 or (vc.ndim == 2 and len(vc) != nv):
                color = vc if vc.ndim == 1 else vc[0]
                mesh_list[i].visual.vertex_colors = np.tile(color, (nv, 1))

        combined_mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]

        pc_size = self.pc_size
        sampled_points, face_idx, sampled_colors = trimesh.sample.sample_surface(
            combined_mesh, pc_size, seed=None, sample_color=True
        )


        return sampled_points, sampled_colors, face_idx, combined_mesh, mesh_list

    @staticmethod
    def _build_obj_surface(sampled_points, sampled_rgb):
        """
        Build obj_surface tensor from raw point cloud data.

        Args:
            sampled_points: [N, 3] numpy float array (xyz)
            sampled_rgb: [N, 3] numpy float array (rgb in [0, 1])

        Returns:
            obj_surface: [1, N, 7] torch.FloatTensor (xyz + rgb + sharpedge)
        """
        pc_size = sampled_points.shape[0]
        sharpedge_label = np.zeros((pc_size, 1), dtype=np.float32)
        obj_surface = torch.FloatTensor(
            np.concatenate([sampled_points, sampled_rgb, sharpedge_label], axis=-1)
        )
        obj_surface = obj_surface.unsqueeze(0)
        return obj_surface

    def _build_face_to_part_masks(self, face_idx, masks_json, mesh_face_num_json, target_mask_ids):
        """
        Build point-level segmentation masks for each target part.

        Uses annotation's mesh_face_num as face offsets (consistent with official PartNeXt code).
        face_idx comes from sampling on the concatenated mesh (reordered to match annotation).

        Args:
            face_idx: [pc_size] face indices into the concatenated (reordered) mesh
            masks_json: JSON string of {part_id: {mesh_idx: [face_indices]}}
            mesh_face_num_json: JSON string of {mesh_idx: num_faces}
            target_mask_ids: list of list of int (per-part maskId groups)

        Returns:
            target_part_masks: [num_parts, pc_size] int64 tensor
        """
        masks_dict = ast.literal_eval(masks_json) if isinstance(masks_json, str) and masks_json else (masks_json or {})
        mesh_face_num = ast.literal_eval(mesh_face_num_json) if isinstance(mesh_face_num_json, str) else mesh_face_num_json


        n_meshes = len(mesh_face_num)
        face_offsets = []
        offset = 0
        for i in range(n_meshes):
            face_offsets.append(offset)
            offset += mesh_face_num[str(i)]

        pc_size = len(face_idx)
        num_parts = len(target_mask_ids)
        target_part_masks = torch.zeros((num_parts, pc_size), dtype=torch.int64)

        for part_idx, mask_id_group in enumerate(target_mask_ids):

            part_face_indices = []
            for mid in mask_id_group:
                mid_str = str(mid)


                if not masks_dict and mid_str in mesh_face_num:
                    mesh_idx = int(mid_str)
                    start = face_offsets[mesh_idx]
                    part_face_indices.append(
                        np.arange(start, start + mesh_face_num[mid_str], dtype=np.int64)
                    )
                    continue
                if mid_str not in masks_dict:
                    continue
                for mesh_idx_str, local_face_indices in masks_dict[mid_str].items():
                    mesh_idx = int(mesh_idx_str)
                    if mesh_idx < len(face_offsets):
                        global_indices = np.atleast_1d(np.array(local_face_indices)) + face_offsets[mesh_idx]
                        part_face_indices.append(global_indices)

            if part_face_indices:
                part_face_indices = np.concatenate(part_face_indices)
                mask = np.isin(face_idx, part_face_indices)
                target_part_masks[part_idx] = torch.from_numpy(mask.astype(np.int64))

        return target_part_masks


    # Message building and tokenization (adapted from Point3DDataset)


    def _build_messages(self, example: dict):
        """
        Build messages with point cloud data injected.

        Returns:
            messages: list of message dicts with point cloud content
            diffusion_model_inputs: dict with vertex, classification_labels, timesteps
            cached_mesh_info: dict with Utonia data if applicable
        """
        messages = example[self.messages_key]
        diffusion_messages = example[self.diffusion_messages_key]
        glb_path = example["glb_path"]
        mesh_face_num_json = example.get("mesh_face_num_json")

        cached_mesh_info = {}


        sampled_points, sampled_colors, face_idx, combined_mesh, mesh_list = (
            self._load_and_sample_point_cloud(glb_path, mesh_face_num_json=mesh_face_num_json)
        )


        _normal = combined_mesh.face_normals[face_idx]
        _normal = _normal / (np.linalg.norm(_normal, axis=-1, keepdims=True) + 1e-8)
        _normal = np.array(_normal, dtype=np.float32)


        _aug_enabled = self.is_train and self.aug_config.get("enabled", False)
        if _aug_enabled:
            _coord = np.array(sampled_points, dtype=np.float32)
            _color_rgb = sampled_colors[:, :3].copy()
            _coord, _color_rgb, _normal, _transform_fn = augment_point_cloud(
                _coord, _color_rgb, _normal, self.aug_config
            )
            sampled_points = _coord
            sampled_colors = np.concatenate([_color_rgb, sampled_colors[:, 3:4]], axis=1)
        else:
            _transform_fn = lambda pts: pts  # noqa: E731


        original_bboxes = example.get("target_bboxes")
        if original_bboxes is not None and _aug_enabled:
            transformed_bboxes = []
            for bbox in original_bboxes:
                if isinstance(bbox, str):
                    bbox = ast.literal_eval(bbox)
                bbox_float = [float(v) for v in bbox]
                transformed_bboxes.append(transform_bbox_aabb(bbox_float, _transform_fn))
        else:
            transformed_bboxes = original_bboxes


        sampled_rgb = sampled_colors[:, :3].astype(np.float32) / 255.0
        obj_surface = self._build_obj_surface(sampled_points, sampled_rgb)

        cached_mesh_info["__sampled_points"] = sampled_points
        cached_mesh_info["__face_idx"] = face_idx
        cached_mesh_info["__combined_mesh"] = combined_mesh
        cached_mesh_info["__obj_surface"] = obj_surface


        num_components, comp_ids = igl.facet_components(combined_mesh.faces)
        max_part_num = 100
        part_name_to_id = {}
        for comp_idx in range(num_components):
            if np.sum(comp_ids == comp_idx) < 1:
                continue
            part_name_to_id[comp_idx] = random.randint(0, max_part_num - 1)

        cached_mesh_info["__comp_ids"] = comp_ids
        cached_mesh_info["__part_name_to_id"] = part_name_to_id


        _coord = np.array(sampled_points, dtype=np.float32)
        _color = sampled_colors[:, :3]
        utonia_pc = dict(coord=_coord, color=_color, normal=_normal)
        utonia_point = self.utonia_transform(utonia_pc)
        cached_mesh_info["__utonia_point"] = utonia_point

        prompt_mode = example.get("prompt_mode", "grounding")

        if prompt_mode == "grounding":
            target_part_names = example["target_part_names"]
            target_mask_ids = example["target_mask_ids"]
            target_bboxes = transformed_bboxes

            n = len(target_part_names)
            if n > 0 and target_bboxes is not None:

                unique_names = list(dict.fromkeys(target_part_names))
                m = len(unique_names)
                k = random.randint(1, m)
                sampled_names = random.sample(unique_names, k)
                sampled_names_set = set(sampled_names)


                sampled_indices = [i for i in range(n) if target_part_names[i] in sampled_names_set]


                sampled_unique = list(dict.fromkeys(target_part_names[i] for i in sampled_indices))
                user_content = f"Please segment all the {'; '.join(sampled_unique)} in <point_cloud>."

                assistant_lines = []
                new_part_names = []
                new_mask_ids = []
                for i in sampled_indices:
                    bbox = target_bboxes[i]
                    if isinstance(bbox, str):
                        bbox = ast.literal_eval(bbox)
                    bbox_list = [round(float(v), 3) for v in bbox]
                    bbox_json = json.dumps({"bbox_aabb": bbox_list, "label": target_part_names[i]})
                    assistant_lines.append(f"{bbox_json}<|SEG|>")
                    new_part_names.append(target_part_names[i])
                    new_mask_ids.append(target_mask_ids[i])

                assistant_text = "\n".join(assistant_lines)
                assistant_text += "\n<|BG|>"
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_text},
                ]
                example["target_part_names"] = new_part_names
                example["target_mask_ids"] = new_mask_ids

        elif prompt_mode == "promptable":
            target_part_names = example["target_part_names"]
            target_mask_ids = example["target_mask_ids"]
            target_bboxes = transformed_bboxes
            masks_json = example["masks_json"]
            mesh_face_num_json_str = example["mesh_face_num_json"]


            all_masks = self._build_face_to_part_masks(
                face_idx, masks_json, mesh_face_num_json_str, target_mask_ids
            )
            num_pos_per_part = all_masks.sum(dim=1)
            valid_indices = torch.nonzero(num_pos_per_part > 0, as_tuple=False).squeeze(1).reshape(-1).tolist()

            if not valid_indices:

                sampled_points, sampled_colors, face_idx, combined_mesh, mesh_list = (
                    self._load_and_sample_point_cloud(glb_path, mesh_face_num_json=mesh_face_num_json)
                )

                _normal = combined_mesh.face_normals[face_idx]
                _normal = _normal / (np.linalg.norm(_normal, axis=-1, keepdims=True) + 1e-8)
                _normal = np.array(_normal, dtype=np.float32)

                if _aug_enabled:
                    _coord = np.array(sampled_points, dtype=np.float32)
                    _color_rgb = sampled_colors[:, :3].copy()
                    _coord, _color_rgb, _normal, _transform_fn = augment_point_cloud(
                        _coord, _color_rgb, _normal, self.aug_config
                    )
                    sampled_points = _coord
                    sampled_colors = np.concatenate([_color_rgb, sampled_colors[:, 3:4]], axis=1)

                if original_bboxes is not None and _aug_enabled:
                    transformed_bboxes = []
                    for bbox in original_bboxes:
                        if isinstance(bbox, str):
                            bbox = ast.literal_eval(bbox)
                        bbox_float = [float(v) for v in bbox]
                        transformed_bboxes.append(transform_bbox_aabb(bbox_float, _transform_fn))
                    target_bboxes = transformed_bboxes

                sampled_rgb = sampled_colors[:, :3].astype(np.float32) / 255.0
                obj_surface = self._build_obj_surface(sampled_points, sampled_rgb)
                cached_mesh_info["__sampled_points"] = sampled_points
                cached_mesh_info["__face_idx"] = face_idx
                cached_mesh_info["__combined_mesh"] = combined_mesh
                cached_mesh_info["__obj_surface"] = obj_surface

                _coord = np.array(sampled_points, dtype=np.float32)
                _color = sampled_colors[:, :3]
                utonia_pc = dict(coord=_coord, color=_color, normal=_normal)
                utonia_point = self.utonia_transform(utonia_pc)
                cached_mesh_info["__utonia_point"] = utonia_point
                all_masks = self._build_face_to_part_masks(
                    face_idx, masks_json, mesh_face_num_json_str, target_mask_ids
                )
                num_pos_per_part = all_masks.sum(dim=1)
                valid_indices = torch.nonzero(num_pos_per_part > 0, as_tuple=False).squeeze(1).reshape(-1).tolist()

            if not valid_indices:

                logger.warning(
                    f"[promptable] No valid parts after resample for {example.get('model_id', '?')}, "
                    f"falling back to open mode"
                )

                assistant_lines = []
                for i in range(len(target_part_names)):
                    bbox = target_bboxes[i]
                    if isinstance(bbox, str):
                        bbox = ast.literal_eval(bbox)
                    bbox_list = [round(float(v), 3) for v in bbox]
                    bbox_json = json.dumps({"bbox_aabb": bbox_list, "label": target_part_names[i]})
                    assistant_lines.append(f"{bbox_json}<|SEG|>")
                fallback_assistant_text = "\n".join(assistant_lines)
                fallback_assistant_text += "\n<|BG|>"
                messages = [
                    {"role": "user", "content": "Please segment all parts of this 3D object in <point_cloud>."},
                    {"role": "assistant", "content": fallback_assistant_text},
                ]
            else:
                part_idx = random.choice(valid_indices)
                part_mask = all_masks[part_idx]
                num_pos = int(num_pos_per_part[part_idx].item())
                part_name = target_part_names[part_idx]
                part_bbox = target_bboxes[part_idx]
                if isinstance(part_bbox, str):
                    part_bbox = ast.literal_eval(part_bbox)


                pos_indices = torch.nonzero(part_mask > 0, as_tuple=False).squeeze(1).numpy()
                pos_points = sampled_points[pos_indices]
                centroid = pos_points.mean(axis=0)
                dists = np.linalg.norm(pos_points - centroid, axis=1)
                k = min(max(10, num_pos // 3), num_pos)
                if k >= num_pos:
                    top_k_local = np.arange(num_pos)
                else:
                    top_k_local = np.argpartition(dists, k - 1)[:k]
                chosen_local = top_k_local[random.randint(0, k - 1)]
                prompt_point = pos_points[chosen_local]


                point_str = f"({prompt_point[0]:.3f}, {prompt_point[1]:.3f}, {prompt_point[2]:.3f})"


                user_content = f"Please segment the part at {point_str} in <point_cloud>."
                part_bbox_list = [round(float(v), 3) for v in part_bbox]
                bbox_json = json.dumps({"bbox_aabb": part_bbox_list, "label": part_name})
                assistant_text = f"{bbox_json}<|SEG|>"
                assistant_text += "\n<|BG|>"
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_text},
                ]
                example["target_part_names"] = [part_name]
                example["target_mask_ids"] = [target_mask_ids[part_idx]]
        elif prompt_mode == "open":
            granularity = example.get("granularity")
            if granularity is None:
                raise ValueError(
                    f"[open mode] Missing 'granularity' field for model_id={example.get('model_id', '?')}. "
                    f"Please regenerate parquet with updated prepare_partnext_dataset.py."
                )
            num_parts = len(example["target_part_names"])


            if random.random() < OPEN_GENERIC_PROB:
                user_content = random.choice(OPEN_GENERIC_TEMPLATES)
            else:
                pool = OPEN_TEMPLATES[granularity]
                if random.random() < pool["weights"][0] and pool["semantic"]:
                    template = random.choice(pool["semantic"])
                    user_content = template
                else:
                    template = random.choice(pool["numeric"])
                    noisy_n = max(2, num_parts + random.choice([-1, 0, 0, 0, 1]))
                    user_content = template.format(n=noisy_n)


            target_part_names_open = example["target_part_names"]
            target_bboxes_open = transformed_bboxes
            if target_bboxes_open is not None and len(target_bboxes_open) == len(target_part_names_open):
                assistant_lines_open = []
                for i in range(len(target_part_names_open)):
                    bbox = target_bboxes_open[i]
                    if isinstance(bbox, str):
                        bbox = ast.literal_eval(bbox)
                    bbox_list = [round(float(v), 3) for v in bbox]
                    bbox_json = json.dumps({"bbox_aabb": bbox_list, "label": target_part_names_open[i]})
                    assistant_lines_open.append(f"{bbox_json}<|SEG|>")
                assistant_text = "\n".join(assistant_lines_open)
            else:
                assistant_text = messages[1]["content"] if len(messages) > 1 else ""
            if "<|BG|>" not in assistant_text:
                assistant_text += "\n<|BG|>"
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_text},
            ]


        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_msg is not None:
            content = assistant_msg["content"] if isinstance(assistant_msg["content"], str) else ""
            if "<|BG|>" not in content:
                logger.warning(
                    f"Assistant message is missing <|BG|> for {example.get('model_id', '?')}"
                )


        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue
            if self.processor is None:
                continue

            content_list = []
            segments = re.split("(<point_cloud>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<point_cloud>":
                    content_list.append({"type": "point_cloud", "point_cloud": obj_surface.to(torch.bfloat16)})
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list


        diffusion_model_inputs = {}
        if diffusion_messages.get("targets_type") == "classification":
            target_mask_ids = example["target_mask_ids"]
            masks_json = example["masks_json"]
            mesh_face_num_json = example["mesh_face_num_json"]

            target_part_masks = self._build_face_to_part_masks(
                face_idx, masks_json, mesh_face_num_json, target_mask_ids
            )


            pc_size = sampled_points.shape[0]
            sampled_comp_ids = comp_ids[face_idx]
            part_id_to_point_mask = torch.zeros((pc_size, max_part_num), dtype=torch.bool)
            for comp_idx, part_id in part_name_to_id.items():
                mask = sampled_comp_ids == comp_idx
                part_id_to_point_mask[mask, part_id] = True

            model_input = obj_surface.squeeze(0)
            diffusion_model_inputs["vertex"] = torch.cat(
                [model_input.to(torch.bfloat16), part_id_to_point_mask.to(torch.bfloat16)], dim=1
            )
            diffusion_model_inputs["timesteps"] = torch.zeros((1,))
            S = target_part_masks.shape[0]
            has_part = target_part_masks.any(dim=0)
            classification_labels = target_part_masks.argmax(dim=0)
            classification_labels[~has_part] = S
            diffusion_model_inputs["classification_labels"] = classification_labels

        return messages, diffusion_model_inputs, cached_mesh_info

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Tokenize a single message. Adapted from Point3DDataset."""
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = processor.apply_chat_template(
            [message],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]


        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt):]
            attention_mask = attention_mask[len(self.system_prompt):]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            loss_mask[:len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], enable_thinking: bool):
        """Verify concatenated input_ids match whole-sequence tokenization."""
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "PartNeXtPoint3DDataset: apply_chat_template per-turn concat != whole-sequence. "
            "Set `ignore_input_ids_mismatch=True` to ignore."
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning(error_message)
            else:
                raise AssertionError(error_message)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages, diffusion_model_inputs, cached_mesh_info = self._build_messages(row_dict)


        from recipe.part_seg.eval_metrics import PROMPT_MODE_TO_ID
        prompt_mode = row_dict.get("prompt_mode", "grounding")
        _prompt_mode_id = torch.tensor([PROMPT_MODE_TO_ID.get(prompt_mode, 0)], dtype=torch.long)


        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            try:
                _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                    index=i,
                    message=message,
                )
                input_ids.append(_input_ids)
                loss_mask.append(_loss_mask)
                attention_mask.append(_attention_mask)
                for k, v in _inputs.items():
                    multi_modal_inputs.setdefault(k, []).append(v)
            except Exception as e:
                print(f"[PartNeXt] Error processing message {i}: {e}")
                raise

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape

        self.sanity_check(input_ids, messages, enable_thinking=None)


        for k, v in diffusion_model_inputs.items():
            multi_modal_inputs.setdefault(k, []).append(v)


        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)
        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                multi_modal_inputs[k] = torch.cat(v, dim=0)


        multi_modal_inputs["prompt_mode_id"] = _prompt_mode_id.unsqueeze(0)


        from transformers.models.qwen3_vl import utonia as _utonia
        if "__utonia_point" in cached_mesh_info:
            utonia_point_dict = _utonia.data.collate_fn([cached_mesh_info["__utonia_point"]])
            multi_modal_inputs["utonia_point_dict"] = utonia_point_dict


        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)


        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                    loss_mask = loss_mask[-self.max_length:]
                    position_ids = position_ids[..., -self.max_length:]
                elif self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                    position_ids = position_ids[..., :self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            position_ids = position_ids.transpose(0, 1)
            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                position_ids = position_ids[..., :self.max_length]

            position_ids = position_ids.transpose(0, 1)
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
