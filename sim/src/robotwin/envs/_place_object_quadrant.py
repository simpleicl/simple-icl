from simpleicl.paths import ASSETS
import glob
import os

import numpy as np
import sapien

from ._base_task import Base_Task
from ._ood_eval_utils import attach_actor_to_arm
from .scene.task_requirements import GRASPABLE_SMALL
from .utils import *


class PlaceObjectQuadrant(Base_Task):
    target_xy = (0.0, 0.0)

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    @staticmethod
    def _model_ids(modelname):
        paths = glob.glob(
            os.path.join(f"{ASSETS}/objects", modelname, "model_data*.json")
        )
        return sorted(int(os.path.basename(path)[10:-5]) for path in paths)

    def load_actors(self):
        self.object_name = str(np.random.choice(sorted(GRASPABLE_SMALL)))
        self.object_id = int(np.random.choice(self._model_ids(self.object_name)))
        source_x = [0.18, 0.27] if self.target_xy[0] < 0 else [-0.27, -0.18]
        source_y = [-0.16, -0.06] if self.target_xy[1] > 0 else [0.05, 0.14]
        self.object = create_actor(
            scene=self,
            pose=rand_pose(
                xlim=source_x,
                ylim=source_y,
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi],
            ),
            modelname=self.object_name,
            model_id=self.object_id,
            convex=True,
        )
        self._configure_object()
        self._create_pad()

    def bind_actors(self, actors_by_role):
        self.object = actors_by_role["object"]
        self.object_name = self.object.get_name().split("#")[0]
        self.object_id = getattr(self.object, "model_id", 0)
        self._configure_object()
        self._create_pad()

    def _configure_object(self):
        self.object.set_mass(0.03)
        self.add_prohibit_area(self.object, padding=0.06)

    def _create_pad(self):
        self.pad = create_box(
            scene=self.scene,
            pose=sapien.Pose(
                [self.target_xy[0], self.target_xy[1], 0.741 + self.table_z_bias],
                [1, 0, 0, 0],
            ),
            half_size=[0.055, 0.05, 0.0005],
            color=(0.1, 0.35, 1.0),
            is_static=True,
            name="quadrant_pad",
        )
        self.add_prohibit_area(self.pad, padding=0.08)

    def play_once(self):
        arm_tag = ArmTag("left" if self.object.get_pose().p[0] < 0 else "right")
        # Preserve the settled support height of meshes whose origin is above
        # their contact plane (for example, mouse and stapler assets).
        support_offset = max(
            0.0,
            float(self.object.get_pose().p[2] - (0.741 + self.table_z_bias)),
        )
        self._expert_stage = "grasp"
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        # Scene objects are dynamic; keep the grasp rigid while transferring
        # them so they cannot lag behind the gripper or gain release velocity.
        drive = attach_actor_to_arm(self, self.object, arm_tag)
        self._expert_stage = "lift"
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        pad_pos = self.pad.get_pose().p
        target_pose = [
            float(pad_pos[0]),
            float(pad_pos[1]),
            float(pad_pos[2] + support_offset),
        ] + self.object.get_pose().q.tolist()
        self._expert_stage = "place"
        self.move(
            self.place_actor(
                self.object,
                arm_tag=arm_tag,
                target_pose=target_pose,
                constrain="free",
                pre_dis=0.08,
                dis=0.0,
                is_open=False,
            )
        )
        self._expert_stage = "release"
        self.move(self.open_gripper(arm_tag))
        drive.disable()
        self.info["info"] = {
            "{A}": f"{self.object_name}/base{self.object_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pos = self.object.get_pose().p
        pad_pos = self.pad.get_pose().p
        self._success_checks = {
            "xy": np.all(
                np.abs(object_pos[:2] - pad_pos[:2]) < np.array([0.045, 0.04])
            ),
            "height": object_pos[2] > pad_pos[2] - 0.015,
            "grippers": self.is_left_gripper_open() and self.is_right_gripper_open(),
        }
        return all(self._success_checks.values())
