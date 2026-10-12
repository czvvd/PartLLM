"""Built-in PartLLM segmentation prompts."""

SEMANTIC = {
    "coarse": "Please segment and name the main parts of this 3D object in <point_cloud>.",
    "general": "Please segment and name all parts of this 3D object in <point_cloud>.",
    "fine": "Please segment and name all the detailed parts of this 3D object in <point_cloud>.",
}

NON_SEMANTIC = {
    "coarse": "Please segment this 3D object into its main geometric parts in <point_cloud>.",
    "general": "Please segment this 3D object into geometric parts in <point_cloud>.",
    "fine": "Please segment this 3D object into detailed geometric parts in <point_cloud>.",
}


def select_prompts(task: str, sem_mode: str, prompt_set: str, part_names=None, num_parts=None):
    if task == "text_guided":
        names = list(dict.fromkeys(name.strip() for name in part_names or [] if name.strip()))
        if not names:
            raise ValueError("Text-guided segmentation requires --part_names")
        return [("text_guided", f"Please segment all the {'; '.join(names)} in <point_cloud>.")]
    if num_parts is not None and num_parts <= 0:
        raise ValueError("--num_parts must be positive")
    modes = ("semantic", "nosem") if sem_mode == "all" else (sem_mode,)
    selected = ("coarse", "general", "fine") if prompt_set == "all" else (prompt_set,)
    result = []
    for mode in modes:
        prefix = "" if mode == "semantic" else "nosem_"
        if num_parts is not None:
            template = (
                f"Please segment and name about {num_parts} parts of this 3D object in <point_cloud>."
                if mode == "semantic" else
                f"Please segment this 3D object into about {num_parts} geometric parts in <point_cloud>."
            )
            result.append((f"{prefix}num_parts_{num_parts}", template))
        else:
            templates = SEMANTIC if mode == "semantic" else NON_SEMANTIC
            result.extend((prefix + name, templates[name]) for name in selected)
    return result
