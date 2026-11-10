"""Read archived OOD scenes using the public task names.

Aliases apply only to persisted metadata; simulator modules use public names.
Keep the original sampling keys so a rename preserves the seed-42 benchmark.
"""

LEGACY_TASK_NAMES = {
    "place_toycar_plasticbox": "recombine_place_toycar_plasticbox",
    "place_cup_basket": "recombine_place_cup_basket",
    "touch_toycar": "recombine_touch_toycar",
    "place_playingcards_plate": "recombine_place_playingcards_plate",
    "place_block_pad": "recombine_place_block_pad",
    "place_mouse_bowl": "recombine_place_mouse_bowl",
}
_PUBLIC_TASK_NAMES = {legacy: public for public, legacy in LEGACY_TASK_NAMES.items()}


def public_task_name(name: str) -> str:
    return _PUBLIC_TASK_NAMES.get(name, name)


def sampling_task_name(name: str) -> str:
    return LEGACY_TASK_NAMES.get(public_task_name(name), name)


def sampling_scene_name(name: str) -> str:
    """Preserve episode selection when an archived scene directory is renamed."""
    task_name, separator, code = name.partition("__")
    if not separator:
        return name
    return f"{sampling_task_name(task_name)}__{code}"


def public_scene_metadata(value):
    """Translate exact task identifiers, including dictionary keys, in memory."""
    if isinstance(value, dict):
        return {
            public_task_name(key): public_scene_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_scene_metadata(item) for item in value]
    if isinstance(value, str):
        return public_task_name(value)
    return value
