import numpy as np

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from ._base_task import Base_Task
from .utils import *


class close_laptop(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _configure(self):
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.8])
        self.laptop.set_mass(0.01)
        self.laptop.set_properties(1, 0)
        self.add_prohibit_area(self.laptop, padding=0.1)

    def load_actors(self):
        self.model_name = "015_laptop"
        self.model_id = int(np.random.randint(0, 11))
        self.laptop = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        self._configure()

    def bind_actors(self, actors_by_role):
        self.laptop = actors_by_role["laptop"]
        self.model_name = "015_laptop"
        self.model_id = getattr(self.laptop, "model_id", 0)
        self._configure()

    def play_once(self):
        self.arm_tag = ArmTag("left")
        quat = GRASP_DIRECTION_DIC[
            "top_down_little_right"
            if self.arm_tag == "left"
            else "top_down_little_left"
        ]
        self.move(self.close_gripper(self.arm_tag, pos=0.55))
        limit = self.laptop.get_qlimits()[0]
        target = limit[0] + (limit[1] - limit[0]) * 0.3
        contact = np.asarray(self.laptop.get_contact_point(1)[:3], dtype=float)
        self.move(
            self.move_to_pose(
                self.arm_tag,
                [contact[0], contact[1], contact[2] + 0.1] + quat,
            )
        )
        self.move(
            self.move_to_pose(
                self.arm_tag,
                [contact[0], contact[1], contact[2] + 0.02] + quat,
            )
        )
        for _ in range(8):
            contact = np.asarray(self.laptop.get_contact_point(1)[:3], dtype=float)
            self.move(
                self.move_to_pose(
                    self.arm_tag,
                    [contact[0], contact[1] + 0.025, contact[2] + 0.02] + quat,
                )
            )
            if not self.plan_success or float(self.laptop.get_qpos()[0]) <= target:
                break
        self.move(self.open_gripper(self.arm_tag))
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        limit = self.laptop.get_qlimits()[0]
        qpos = float(self.laptop.get_qpos()[0])
        return qpos <= limit[0] + (limit[1] - limit[0]) * 0.3 and self.plan_success
