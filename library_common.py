"""
Helpers shared by the Media Library, Reviewer and Compilation Planner pages.
"""


def scene_tags(scene: dict) -> list:
    """The tags to use for a scene: the Reviewer's corrected_tags when the scene
    has been corrected (even if the correction removed every tag), otherwise the
    auto-generated tags."""
    if "corrected_tags" in scene and scene["corrected_tags"] is not None:
        return list(scene["corrected_tags"])
    return list(scene.get("tags", []))
