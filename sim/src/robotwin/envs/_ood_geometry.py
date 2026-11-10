"""Collision-mesh regions for placement and visible-surface contact."""

import numpy as np
import sapien
import trimesh
import shapely
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import unary_union


def collision_mesh(actor):
    body = actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        body = actor.actor.find_component_by_type(
            sapien.physx.PhysxRigidStaticComponent
        )
    if body is None:
        body = actor.actor.find_component_by_type(
            sapien.physx.PhysxArticulationLinkComponent
        )
    if body is None:
        raise ValueError("Actor has no physical collision body")
    meshes = []
    for shape in body.get_collision_shapes():
        if isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * np.asarray(shape.half_size))
        else:
            mesh = trimesh.Trimesh(
                np.asarray(shape.vertices) * np.asarray(shape.scale),
                np.asarray(shape.triangles),
                process=False,
            )
        mesh.apply_transform(shape.local_pose.to_transformation_matrix())
        meshes.append(mesh)
    if not meshes:
        raise ValueError("Missing collision mesh")
    return trimesh.util.concatenate(meshes)


def transformed_points(mesh, pose):
    return trimesh.transform_points(mesh.vertices, pose.to_transformation_matrix())


def upper_surface(mesh, xy):
    origins = np.column_stack([xy, np.full(len(xy), mesh.bounds[1, 2] + 0.1)])
    locations, rays, _ = mesh.ray.intersects_location(
        origins, np.tile([0, 0, -1], (len(xy), 1))
    )
    heights = np.full(len(xy), -np.inf)
    np.maximum.at(heights, rays, locations[:, 2])
    return heights


class PlacementRegion:
    STEP = 0.003

    def __init__(self, mesh, kind):
        self.kind = kind
        low, high = mesh.bounds
        x = np.arange(low[0] + self.STEP, high[0], self.STEP)
        y = np.arange(low[1] + self.STEP, high[1], self.STEP)
        self.xy = np.array(np.meshgrid(x, y)).reshape(2, -1).T
        heights = upper_surface(mesh, self.xy)
        self.rim = {"bowl": 0.047, "basket": 0.085, "tray": 0.020}.get(kind)
        valid = np.isfinite(heights)
        if kind in {"plate", "skillet", "plasticbox"}:
            # Infer the cavity floor from the original collision surface.
            floor = (
                float(np.quantile(heights[valid], 0.25))
                if valid.any()
                else float("nan")
            )
            self.rim = float(high[2])
            valid &= (heights <= floor + 0.006) & (heights < self.rim - 0.008)
        if self.rim is not None:
            valid &= heights < self.rim - 0.008
        else:
            valid &= heights > high[2] - 0.006
        self.xy, self.heights = self.xy[valid], heights[valid]
        if not len(self.xy):
            raise ValueError(f"No accessible support surface: {kind}")
        h = self.STEP / 2 + 1e-7
        self.polygon = unary_union(
            [box(a - h, b - h, a + h, b + h) for a, b in self.xy]
        ).buffer(-0.002)
        if self.polygon.is_empty:
            raise ValueError(f"No usable interior: {kind}")
        # The basket handle occludes its convex floor, but does not divide the cavity.
        self.interior = self.polygon.convex_hull if kind == "basket" else self.polygon

    def contains(self, points):
        footprint = MultiPoint(points[:, :2]).convex_hull
        return bool(
            self.interior.covers(footprint)
            and (self.rim is None or points[:, 2].min() < self.rim - 0.004)
        )

    def destination(self, points):
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        footprint = MultiPoint(points[:, :2] - center[:2]).convex_hull
        candidates = shapely.polygons(
            np.asarray(footprint.exterior.coords)[None, :, :] + self.xy[:, None, :]
        )
        for region in (self.polygon, self.interior):
            fits = np.flatnonzero(shapely.covers(region, candidates))
            if len(fits):
                margin = shapely.distance(region.boundary, candidates[fits])
                centrality = np.linalg.norm(
                    self.xy[fits] - self.xy.mean(axis=0), axis=1
                )
                score = margin - 1e-6 * centrality
                if region is not self.polygon:
                    visible = shapely.area(
                        shapely.intersection(self.polygon, candidates[fits])
                    )
                    score = visible + 1e-5 * score
                idx = fits[np.argmax(score)]
                return np.array([*self.xy[idx], self.heights[idx]])
        raise ValueError("Object footprint does not fit the unmodified target")

    def oriented_destination(self, points):
        def long_axis(polygon):
            corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)
            edges = np.diff(corners, axis=0)
            edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
            return np.arctan2(edge[1], edge[0])

        aligned = long_axis(self.polygon) - long_axis(
            MultiPoint(points[:, :2]).convex_hull
        )
        aligned = (aligned + np.pi / 2) % np.pi - np.pi / 2
        angles = (aligned, 0) if self.kind == "pad" else (0, aligned)
        for angle in (
            *angles,
            *[
                v
                for step in range(1, 7)
                for v in (step * np.pi / 12, -step * np.pi / 12)
            ],
        ):
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            rotated = points.copy()
            rotated[:, :2] = points[:, :2] @ rotation.T
            try:
                return self.destination(rotated), angle
            except ValueError:
                continue
        raise ValueError(
            "Object footprint does not fit the unmodified target at any tested yaw"
        )


def touch_point(mesh):
    low, high = mesh.bounds
    xy = (
        np.array(
            np.meshgrid(
                np.linspace(low[0], high[0], 17)[1:-1],
                np.linspace(low[1], high[1], 17)[1:-1],
            )
        )
        .reshape(2, -1)
        .T
    )
    heights = upper_surface(mesh, xy)
    valid = np.isfinite(heights) & (heights >= high[2] - 0.008)
    if not valid.any():
        raise ValueError("No visible upper contact surface")
    xy, heights = xy[valid], heights[valid]
    idx = np.argmin(np.linalg.norm(xy - (low[:2] + high[:2]) / 2, axis=1))
    return np.array([*xy[idx], heights[idx]])
