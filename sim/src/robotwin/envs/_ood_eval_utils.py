"""Physical contact and stability checks for the OOD benchmark."""

import numpy as np
import sapien


def open_grippers(task) -> bool:
    return task.is_left_gripper_open() and task.is_right_gripper_open()


def supported_by(actor, support, contacts) -> bool:
    """Require upward contact impulses between an object and its support."""
    ids = {
        actor.actor.per_scene_id,
        getattr(support, "actor", support).per_scene_id,
    }
    return any(
        {body.entity.per_scene_id for body in contact.bodies} == ids
        and any(
            abs(point.normal[2]) > 0.9
            and point.separation <= 0.004
            and abs(point.impulse[2]) > 1e-8
            for point in contact.points
        )
        for contact in contacts
    )


def slow_actor(actor) -> bool:
    """Static actors and sufficiently settled dynamic actors are stable."""
    body = actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    return body is None or (
        np.linalg.norm(body.linear_velocity) <= 0.02
        and np.linalg.norm(body.angular_velocity) <= 0.2
    )


def attach_actor_to_arm(task, actor, arm_tag):
    link = (
        task.robot.left_ee.child_link
        if arm_tag == "left"
        else task.robot.right_ee.child_link
    )
    raw = actor.actor
    link_pose = (
        link.get_entity_pose() if hasattr(link, "get_entity_pose") else link.get_pose()
    )
    drive = task.scene.create_drive(
        link, link_pose.inv() * raw.get_pose(), raw, sapien.Pose()
    )
    for axis in ("x", "y", "z"):
        getattr(drive, f"set_drive_property_{axis}")(2e4, 1e3)
    drive.set_drive_property_slerp(2e4, 1e3)
    return drive
