"""Single-object transfer and touch tasks without grasp assistance."""

import numpy as np
import sapien
from copy import deepcopy
from types import SimpleNamespace

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from ._ood_eval_utils import open_grippers, supported_by, slow_actor
from ._ood_geometry import (
    collision_mesh,
    transformed_points,
    PlacementRegion,
    touch_point,
)
from .scene.ood_catalog import TASKS, OBJECTS, TARGETS, layout
from .utils import ArmTag, UnStableError, create_actor, create_box, rand_pose


class OODTask(Base_Task):
    STABLE_SECONDS = 0.4
    _supported_by = staticmethod(supported_by)
    _slow = staticmethod(slow_actor)

    def setup_demo(self, **kwargs):
        self.spec = TASKS[self.__class__.__name__]
        self.profile = OBJECTS[self.spec["object"]]
        self._success_checks = {}
        self._expert_stage = "initialize"
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        actors = {}
        for role, (model, region) in zip(
            ("object", "target"), layout(self.__class__.__name__)
        ):
            pose = rand_pose(
                xlim=region["xlim"],
                ylim=region["ylim"],
                zlim=[0.76 + self.table_z_bias],
                qpos=region["qpos"],
            )
            if model == "sapien-block":
                box_spec = region["box_spec"]
                actor = create_box(
                    self.scene,
                    pose,
                    half_size=box_spec["half_size"],
                    color=box_spec["color"],
                    name=role,
                )
            else:
                actor = create_actor(
                    self.scene, pose, model, model_id=region["model_id"], convex=True
                )
            actor.model_id = region["model_id"]
            actors[role] = actor
        self.bind_actors(actors)

    def bind_actors(self, actors_by_role):
        self.spec = TASKS[self.__class__.__name__]
        self.profile = OBJECTS[self.spec["object"]]
        self.object = actors_by_role["object"]
        self.target = actors_by_role.get("target")
        if self.spec["object"] == "mouse":
            pose = self.object.get_pose()
            pose.q = self.profile["q"]
            self.object.actor.set_pose(pose)
            for oid, actor in getattr(self, "_scene_actors_by_oid", {}).items():
                if actor is self.object and oid in getattr(
                    self, "_pair_perturbation_offsets", {}
                ):
                    self._pair_perturbation_offsets[oid][2] = 0.0
        model_id = getattr(self.object, "model_id", None)
        if (
            self.profile["model"] != "sapien-block"
            and model_id not in self.profile["ids"]
        ):
            raise ValueError(f"Unsupported source instance: {model_id}")
        if "mass" in self.profile:
            self.object.set_mass(self.profile["mass"])
        if "grasp_contact" in self.profile:
            contact = self.profile["grasp_contact"]
            matrix = np.eye(4)
            matrix[:3, :3] = contact["rotation"]
            matrix[:3, 3] = np.asarray(contact["position"]) / np.asarray(
                self.object.config["scale"]
            )
            self.object.config = deepcopy(self.object.config)
            self.object.config["contact_points_pose"] = [matrix.tolist()]
        if self.target is not None:
            target = TARGETS[self.spec["target"]]
            if "mass" in target:
                self.target.set_mass(target["mass"])
            if target.get("static"):
                pose = self.target.get_pose()
                pose.p = [*pose.p[:2], 0.7405 + self.table_z_bias]
                self.target.actor.set_pose(pose)
                self.target.set_kinematic(True)
        self.arm_tag = ArmTag("left" if self.object.get_pose().p[0] < 0 else "right")
        for actor in actors_by_role.values():
            self.add_prohibit_area(actor, padding=0.03)

    def _frame(self):
        return self.target.get_pose() * self.target_orientation.inv()

    def _points(self, actor):
        return transformed_points(
            self.meshes[actor.actor.per_scene_id], actor.get_pose()
        )

    def _target_points(self):
        return transformed_points(
            self.meshes[self.object.actor.per_scene_id],
            self._frame().inv() * self.object.get_pose(),
        )

    def _on_episode_start(self):
        actors = [self.object] + ([self.target] if self.target is not None else [])
        self.meshes = {a.actor.per_scene_id: collision_mesh(a) for a in actors}
        self.object_start = self.object.get_pose()
        self.max_translation = self.max_rotation = self.stable_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.touch_left_table = False
        self.events = dict(
            approached=False,
            grasped=False,
            transported=False,
            arrived=False,
            contact=False,
            withdrawn=False,
        )
        self.robot_ids = {
            link.entity.per_scene_id
            for entity in (self.robot.left_entity, self.robot.right_entity)
            for link in entity.get_links()
        }
        self.gripper_ids = {
            link.entity.per_scene_id
            for entity in (self.robot.left_entity, self.robot.right_entity)
            for link in entity.get_links()
            if link.entity.name in self.robot.gripper_name
        }
        self.gripper_links = {
            link.entity.per_scene_id: link.entity
            for entity in (self.robot.left_entity, self.robot.right_entity)
            for link in entity.get_links()
            if link.entity.per_scene_id in self.gripper_ids
        }
        self.gripper_meshes = {
            key: collision_mesh(SimpleNamespace(actor=entity))
            for key, entity in self.gripper_links.items()
        }
        if not self.gripper_ids:
            raise ValueError("Cannot identify physical gripper links")
        contacts = self.scene.get_contacts()
        self.initial_valid = bool(
            self._supported_by(self.object, self.table, contacts)
            and self._slow(self.object)
        )
        if self.target is not None:
            self.target_start = self.target.get_pose()
            self.target_orientation = sapien.Pose(q=self.target_start.q)
            mesh = self.meshes[self.target.actor.per_scene_id].copy()
            mesh.apply_transform(self.target_orientation.to_transformation_matrix())
            self.region = PlacementRegion(mesh, self.spec["target"])
            self.destination, self.destination_yaw = self.region.oriented_destination(
                self._target_points()
            )
            self.initial_valid &= not self.region.contains(self._target_points())
        else:
            mesh = self.meshes[self.object.actor.per_scene_id].copy()
            mesh.apply_transform(self.object.get_pose().to_transformation_matrix())
            point = touch_point(mesh)
            self.touch_local = (self.object.get_pose().inv() * sapien.Pose(point)).p
        self.initial_valid &= not self._contacts(contacts)[1]
        self.info["ood"] = dict(
            task=self.__class__.__name__,
            source=self.profile["source"],
            model_id=getattr(self.object, "model_id", 0),
            initial_orientation=self.object_start.q.tolist(),
            evidence="source_allowed_training_instance_unverified",
            events=self.events,
        )
        self._success_checks = dict(initial_valid=self.initial_valid, stable=False)
        if not self.initial_valid:
            raise UnStableError(
                "OOD evaluation requires a settled object on the table outside its target"
            )

    def _contacts(self, contacts):
        obj_id = self.object.actor.per_scene_id
        gripper_points, robot_touching = [], False
        for contact in contacts:
            ids = {b.entity.per_scene_id for b in contact.bodies}
            if obj_id not in ids:
                continue
            for p in contact.points:
                if p.separation <= 0.001:
                    robot_touching |= bool(ids & self.robot_ids)
                    if ids & self.gripper_ids and np.linalg.norm(p.impulse) > 1e-8:
                        gripper_points.append(np.asarray(p.position))
        return gripper_points, robot_touching

    def _after_simulation_step(self):
        dt = self.scene.get_timestep()
        self.elapsed_seconds += dt
        contacts = self.scene.get_contacts()
        points, touching = self._contacts(contacts)
        pose = self.object.get_pose()
        distance = float(np.linalg.norm(pose.p - self.object_start.p))
        self.max_translation = max(self.max_translation, distance)
        angle = 2 * np.arccos(np.clip(abs(np.dot(pose.q, self.object_start.q)), 0, 1))
        self.max_rotation = max(self.max_rotation, float(angle))
        self.events["approached"] |= bool(points)
        supported_table = self._supported_by(self.object, self.table, contacts)
        checks = dict(
            initial_valid=self.initial_valid, slow=bool(self._slow(self.object))
        )
        if self.target is not None:
            lifted = pose.p[2] - self.object_start.p[2] > 0.02 and not supported_table
            self.events["grasped"] |= bool(points and lifted)
            self.events["transported"] |= bool(
                self.events["grasped"]
                and np.linalg.norm(pose.p[:2] - self.object_start.p[:2]) > 0.05
            )
            inside = self.region.contains(self._target_points())
            supported = self._supported_by(self.object, self.target, contacts)
            self.events["arrived"] |= bool(inside and supported)
            checks.update(
                grasped=self.events["grasped"],
                inside=inside,
                target_support=supported,
                released=bool(open_grippers(self) and not touching),
                target_on_table=bool(
                    self.spec["target"] == "pad"
                    or self._supported_by(self.target, self.table, contacts)
                ),
                target_slow=bool(self._slow(self.target)),
                target_upright=bool(
                    self._frame().to_transformation_matrix()[2, 2]
                    >= np.cos(np.deg2rad(15))
                ),
            )
        else:
            self.touch_left_table |= not supported_table
            target_point = (pose * sapien.Pose(self.touch_local)).p
            self.events["contact"] |= any(
                np.linalg.norm(p - target_point) <= 0.018 for p in points
            )
            clearance = float("inf")
            for key, entity in self.gripper_links.items():
                vertices = transformed_points(
                    self.gripper_meshes[key], entity.get_pose()
                )
                nearest = np.clip(
                    target_point, vertices.min(axis=0), vertices.max(axis=0)
                )
                clearance = min(
                    clearance, float(np.linalg.norm(nearest - target_point))
                )
            self.events["withdrawn"] = bool(
                self.events["contact"] and not touching and clearance >= 0.03
            )
            checks.update(
                contact=self.events["contact"],
                withdrawn=self.events["withdrawn"],
                table_support=supported_table,
                never_lifted=not self.touch_left_table,
                preserved=bool(
                    self.max_translation < 0.015 and self.max_rotation < np.deg2rad(8)
                ),
            )
        checks = {k: bool(v) for k, v in checks.items()}
        self.stable_seconds = self.stable_seconds + dt if all(checks.values()) else 0.0
        checks["stable"] = self.stable_seconds >= self.STABLE_SECONDS
        self._success_checks = checks
        self.info["ood"].update(
            checks=checks,
            stable_seconds=self.stable_seconds,
            max_translation=self.max_translation,
            max_rotation=self.max_rotation,
        )

    def play_once(self):
        label = (
            "the red block"
            if self.profile["model"] == "sapien-block"
            else f"{self.profile['model']}/base{getattr(self.object, 'model_id', 0)}"
        )
        self.info["info"] = {"{A}": label}
        if self.target is None:
            self._touch()
        else:
            self._place()
        if self.plan_success:
            self._expert_stage = "stabilize"
            needs_settling = (
                self.spec["object"] == "cup" and self.spec.get("target") == "basket"
            )
            self.delay(3 if needs_settling else 1)
        return self.info

    def _place(self):
        self._expert_stage = "grasp"
        grasp = dict(self.profile["grasp"])
        if self.spec["object"] == "cup":
            self.move(self.close_gripper(self.arm_tag, pos=0.6))
            grasp["contact_point_id"] = 2 if self.arm_tag == "left" else 0
        self.move(self.grasp_actor(self.object, self.arm_tag, **grasp))
        if not self.plan_success:
            return
        self._expert_stage = "lift"
        low = self._points(self.object)[:, 2].min()
        target_top = self._points(self.target)[:, 2].max()
        lift = max(0.06, float(target_top + 0.025 - low))
        self.move(self.move_by_displacement(self.arm_tag, z=lift))
        if not self.plan_success or not self.events["grasped"]:
            return
        self._expert_stage = "orient"
        yaw = sapien.Pose(
            q=[np.cos(self.destination_yaw / 2), 0, 0, np.sin(self.destination_yaw / 2)]
        )
        desired = self._frame() * yaw * sapien.Pose(q=self.object_start.q)
        desired.p = self.object.get_pose().p
        ee = self.get_arm_pose(self.arm_tag)
        goal = desired * self.object.get_pose().inv() * sapien.Pose(ee[:3], ee[3:])
        self.move(self.move_to_pose(self.arm_tag, [*goal.p, *goal.q], speed_scale=0.5))
        if not self.plan_success:
            return
        target = (self._frame() * sapien.Pose(self.destination)).p
        points = self._points(self.object)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        self._expert_stage = "transport"
        self.move(
            self.move_by_displacement(
                self.arm_tag,
                x=float(target[0] - center[0]),
                y=float(target[1] - center[1]),
            )
        )
        if not self.plan_success:
            return
        for _ in range(2):
            points = self._points(self.object)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            delta = target[:2] - center[:2]
            if np.linalg.norm(delta) < 0.0005:
                break
            ee = self.get_arm_pose(self.arm_tag)
            ee[:2] = np.asarray(ee[:2]) + delta
            self.move(self.move_to_pose(self.arm_tag, ee, speed_scale=0.4))
            if not self.plan_success:
                return
        self._expert_stage = "lower"
        release_height = target[2] + 0.003
        if self.spec["object"] == "cup" and self.spec["target"] == "basket":
            # Keep the side grasp above the rim while releasing the cup.
            release_height += 0.08
        self.move(
            self.move_by_displacement(
                self.arm_tag,
                z=float(release_height - self._points(self.object)[:, 2].min()),
            )
        )
        if not self.plan_success:
            return
        self._expert_stage = "release"
        self.move(self.open_gripper(self.arm_tag))
        self.move(self.move_by_displacement(self.arm_tag, z=0.12))

    def _touch(self):
        point = (self.object.get_pose() * sapien.Pose(self.touch_local)).p
        quat = GRASP_DIRECTION_DIC["top_down"]
        self._expert_stage = "approach"
        self.move(self.close_gripper(self.arm_tag))
        self.move(self.move_to_pose(self.arm_tag, [*point[:2], point[2] + 0.25, *quat]))
        if not self.plan_success:
            return
        entity = (
            self.robot.left_entity
            if self.arm_tag == "left"
            else self.robot.right_entity
        )
        joints = getattr(self.robot, f"{self.arm_tag}_gripper")
        finger_ids = {joint.child_link.entity.per_scene_id for joint, *_ in joints}
        names = getattr(self.robot, f"{self.arm_tag}_fix_gripper_name")
        fingers = [
            link.entity
            for link in entity.get_links()
            if link.entity.per_scene_id in finger_ids or link.entity.name in names
        ]
        finger_vertices = [
            transformed_points(
                collision_mesh(SimpleNamespace(actor=link)), link.get_pose()
            )
            for link in fingers
        ]
        vertices = min(finger_vertices, key=lambda points: points[:, 2].min())
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        center = vertices[vertices[:, 2] < low[2] + 0.004].mean(axis=0)
        self.move(
            self.move_by_displacement(
                self.arm_tag,
                x=float(point[0] - center[0]),
                y=float(point[1] - center[1]),
            )
        )
        self._expert_stage = "contact"
        pose = self.get_arm_pose(self.arm_tag)
        pose[2] += float(point[2] - low[2] + 0.003)
        self.move(self.move_to_pose(self.arm_tag, pose, speed_scale=0.25))
        if not self.plan_success:
            return
        for _ in range(36):
            if self.events["contact"] or not self.plan_success:
                break
            pose[2] -= 0.0005
            self.move(self.move_to_pose(self.arm_tag, pose, speed_scale=0.15))
        self._expert_stage = "withdraw"
        self.move(self.move_by_displacement(self.arm_tag, z=0.12))

    def check_success(self):
        return bool(self._success_checks and all(self._success_checks.values()))
