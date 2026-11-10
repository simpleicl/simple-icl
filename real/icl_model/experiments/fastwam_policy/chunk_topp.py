#!/usr/bin/env python3
"""TOPP re-timing of a predicted action chunk (server side).

``server.py`` infers a chunk of ``action_horizon`` raw-qpos waypoints and
queues the first ``replan_steps`` of them, one per client control tick. Nothing
in that path bounds how far apart two consecutive waypoints are, so a chunk that
jumps is commanded as a jump: on the Franka the 1 kHz replay tracker has to chase
it at its own limits.

This module re-times the chunk before it is queued. Because the whole chunk is
in hand at replan time (the horizon is much longer than ``replan_steps`` -- 32 vs
8 in the franka configs) there is a real path to parameterize, so this is proper
TOPP and not a causal filter: no lookahead latency is introduced, we simply use
waypoints the server already has and would otherwise have thrown away.

Pipeline (``ChunkRetimer.retime``)
---------------------------------
1. **Anchor.** The last command handed out is prepended to the chunk, so the
   re-timed path starts exactly where the previous one ended and the command
   stream stays C0-continuous across replans. On the first chunk of an episode
   (or after ``reset``) the measured robot state is used instead.
2. **Geometric smoothing.** A Savitzky-Golay filter over the waypoints. This is
   not cosmetic: TOPP on a raw policy chunk crawls, because per-step noise looks
   like direction reversals that the acceleration limit has to brake for. On a
   representative 32x7 chunk under the franka limits, filtering cuts the re-timed
   duration from 9.5 s to 4.7 s -- same path, half the time. Row 0 (the anchor)
   is pinned so smoothing cannot move the seam.
3. **TOPP.** ``toppra`` parameterizes the cubic spline through those waypoints
   under per-joint velocity and acceleration limits, starting from the path
   speed the previous chunk exited at (projected onto the new path's tangent, so
   speed along the direction we are still going is preserved).
4. **Resample.** The trajectory is sampled at the client's control period, and
   the first ``n_out`` samples are what get queued.
5. **Discrete guard.** The sampled stream is finally passed through a velocity /
   acceleration clamp seeded with the velocity the previous chunk was emitted at.
   This is not redundant with step 3: toppra bounds the trajectory *within* one
   chunk, but consecutive chunks are different paths and it only accepts a scalar
   entry *speed*, not an entry velocity **vector**. Whatever part of the previous
   velocity does not point along the new path is therefore a step at the seam --
   measured at 2.4-2.8x the acceleration limit on a representative rollout, with
   every violation landing exactly on a seam. The clamp is what makes the limits
   hold on the actual emitted command stream; inside a chunk TOPP has already
   done the work and it does not bind.

Re-timing generally makes the chunk *slower* than one-waypoint-per-tick: the
queued ``replan_steps`` samples then cover only part of the path before the
server replans from a fresh observation. That is intended -- the limits are the
hardware's, and the loop is closed, so the next chunk simply picks up from where
the arm actually got to. ``info["time_scale"]`` reports the slowdown so it can
be tuned.

``mode="guard"``
---------------
The catch is that the slowdown is only *intended* while the limits describe the
robot. Feed limits that are tighter than the demonstrations and steps 2-4 have no
way to honour them except by making everything slower, and since only the first
``replan_steps`` samples are ever queued, the arm executes the low-speed head of
every chunk and never reaches the parts that move: the motion comes out smooth
and sluggish and the task fails. Measured on the training data at 30 Hz,
the rollout defaults of 1.0 rad/s and 2.0 rad/s^2 are exceeded by the *recorded
demonstrations* on 24.6% and 81.2% of their own steps respectively (p99: 2.26
rad/s, 15.3 rad/s^2), and 2.0 rad/s^2 over a 33 ms tick permits 0.06 rad/s of
velocity change per step -- 15 ticks to reach 0.9 rad/s from rest, inside an
8-tick execution window. Such a chunk cannot get up to speed before it ends.

``mode="guard"`` is the other operating point: run steps 1 and 5 only. The
chunk's own timing is kept -- one waypoint per control tick, which is the speed
the demonstrations were recorded at -- nothing re-parameterizes time and nothing
filters the geometry, so the path and its speed survive intact. The discrete
clamp is all that runs, and it is a no-op wherever the chunk is already
admissible: it bites at replan seams and on per-step outliers, which is where the
jitter actually comes from, and nowhere else. ``info["guard_bite"]`` /
``info["guard_vel"]`` report how hard it had to work, in both modes.

Pick ``guard`` when ``full`` logs a ``time_scale`` well above 1. That reading
means the limits are tighter than the motion the policy was trained on, and no
amount of tuning inside TOPP will recover the speed.

The joint limits and the control period are the robot's, not the model's, so
they arrive from the client over the ``configure`` command rather than being
configured here (see ``FastWAMPolicy.configure``).

Failure is always soft: any degenerate input or solver failure falls back to the
un-re-timed chunk and reports why in ``info["fallback"]``. The fallback still goes
through the discrete guard, so the limits hold even when TOPP does not run -- a
raw chunk measures ~46x the velocity limit, and the point of this module is that
nothing downstream ever sees that.
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("chunk_topp")

try:
    import toppra as _ta
    import toppra.algorithm as _ta_algo
    import toppra.constraint as _ta_constraint
    from scipy.signal import savgol_filter as _savgol

    # toppra logs one INFO line per solve; a replan every ~0.5 s would drown the
    # server log in them.
    _ta.setup_logging("ERROR")
    TOPP_AVAILABLE = True
    TOPP_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # noqa: BLE001 - optional dependency
    TOPP_AVAILABLE = False
    TOPP_IMPORT_ERROR = _exc

# Velocity/acceleration cap handed to toppra for the gripper dimensions. The
# gripper is a width/opening command, not a joint: it shares neither units nor
# limits with the arm, and delaying an open/close costs a grasp. It still has to
# travel *with* the path (it is a path dimension, so it is interpolated at the
# same path parameter as the joints), it just must never be the constraint that
# slows the path down -- a limit this far above any real gripper travel never
# binds.
_GRIPPER_LIMIT = 1e3

# Below this much joint travel (rad) a chunk is a hold, not a path: the spline
# would be degenerate and there is nothing to re-time.
_MIN_TRAVEL = 1e-6


class ChunkRetimer:
    """Re-time policy action chunks under per-joint velocity/acceleration limits.

    One instance per connected robot; it carries the anchor/velocity continuity
    state across replans, so call ``reset()`` whenever the command stream is
    broken (new episode, or the client dropped the queue to replan).

    Parameters
    ----------
    dt:
        The client's control period (1/fps): the spacing the re-timed trajectory
        is resampled at.
    max_vel, max_acc:
        Per-dimension joint limits over the *full* action width. Entries for
        ``gripper_dims`` are ignored (replaced by ``_GRIPPER_LIMIT``).
    gripper_dims:
        Indices of the gripper entries in the action vector (franka: 7, 15).
    smooth_window, smooth_polyorder:
        Savitzky-Golay geometry filter; ``smooth_window < 3`` disables it.
    limit_margin:
        Limits are scaled by this before being handed to toppra. toppra enforces
        its constraints at the discretization gridpoints, and the cubic spline
        between them overshoots by a few percent (up to ~1.10x acceleration on a
        noisy chunk), so leave headroom rather than emitting samples that breach
        the limit the caller asked for.
    max_anchor_error:
        If the previous command is further than this (rad, worst joint) from the
        measured state, the arm is not tracking; re-seed the anchor from the
        measured state instead of continuing to build on a command the robot
        never reached. ``<= 0`` disables the check.
    mode:
        ``"full"`` runs the whole pipeline. ``"guard"`` runs steps 1 and 5 only,
        keeping the chunk's own timing and geometry and merely clamping the steps
        that break the limits -- see the ``mode="guard"`` section of the module
        docstring for when that is the one you want.
    """

    def __init__(
        self,
        dt: float,
        max_vel: Sequence[float],
        max_acc: Sequence[float],
        gripper_dims: Sequence[int] = (),
        smooth_window: int = 7,
        smooth_polyorder: int = 2,
        limit_margin: float = 0.9,
        max_anchor_error: float = 0.5,
        mode: str = "full",
    ) -> None:
        self.mode = str(mode).lower()
        if self.mode not in ("full", "guard"):
            raise ValueError(f"mode must be 'full' or 'guard', got {mode!r}")
        # Only `full` needs the optional dependencies: `guard` runs steps 1 and 5,
        # which are numpy. Gating both on toppra would make the cheap, no-solver
        # mode unavailable on exactly the machines that cannot install a solver.
        if self.mode == "full" and not TOPP_AVAILABLE:
            raise RuntimeError(
                f"toppra/scipy are required for mode='full': {TOPP_IMPORT_ERROR}"
            )
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be a positive control period, got {dt!r}")

        vel = np.asarray(max_vel, dtype=np.float64).reshape(-1)
        acc = np.asarray(max_acc, dtype=np.float64).reshape(-1)
        if vel.shape != acc.shape:
            raise ValueError(
                f"max_vel {vel.shape} and max_acc {acc.shape} must have the same width"
            )
        self.dof = int(vel.shape[0])
        if self.dof < 1:
            raise ValueError("max_vel/max_acc must be non-empty")

        bad = [int(i) for i in gripper_dims if not 0 <= int(i) < self.dof]
        if bad:
            raise ValueError(
                f"gripper_dims {bad} are out of range for a {self.dof}-dim action"
            )
        self.gripper_dims = tuple(sorted({int(i) for i in gripper_dims}))
        self.joint_dims = np.array(
            [i for i in range(self.dof) if i not in set(self.gripper_dims)], dtype=int
        )
        if self.joint_dims.size == 0:
            raise ValueError("every action dimension was marked as a gripper")

        margin = float(limit_margin)
        if not 0.0 < margin <= 1.0:
            raise ValueError(f"limit_margin must be in (0, 1], got {margin}")
        vel = vel * margin
        acc = acc * margin
        for i in self.gripper_dims:
            vel[i] = acc[i] = _GRIPPER_LIMIT
        if np.any(vel[self.joint_dims] <= 0.0) or np.any(acc[self.joint_dims] <= 0.0):
            raise ValueError("joint max_vel/max_acc entries must all be > 0")
        # toppra wants [dof, 2] (lower, upper) bounds.
        self._vlim = np.stack([-vel, vel], axis=1)
        self._alim = np.stack([-acc, acc], axis=1)

        self.smooth_window = int(smooth_window)
        self.smooth_polyorder = int(smooth_polyorder)
        self.max_anchor_error = float(max_anchor_error)

        # Continuity across replans: the last command handed out, and the joint
        # velocity it was emitted at (the discrete velocity of the command stream,
        # i.e. after the guard in `_enforce_discrete_limits` -- that is what the
        # robot actually got, so it is what the next chunk has to continue from).
        self._last_q: Optional[np.ndarray] = None
        self._last_vel: Optional[np.ndarray] = None

    # ---- description ------------------------------------------------------
    def describe(self) -> str:
        vel = self._vlim[self.joint_dims, 1]
        acc = self._alim[self.joint_dims, 1]
        geom = (
            "chunk timing kept, clamp only" if self.mode == "guard"
            else f"savgol(win={self.smooth_window}, order={self.smooth_polyorder}) + TOPP"
        )
        return (
            f"mode={self.mode} dt={self.dt * 1000:.1f}ms dof={self.dof} "
            f"gripper_dims={list(self.gripper_dims)} "
            f"vel<=[{vel.min():.3f}, {vel.max():.3f}] rad/s "
            f"acc<=[{acc.min():.3f}, {acc.max():.3f}] rad/s^2 "
            f"{geom} (limits already include the margin)"
        )

    def reset(self) -> None:
        """Forget the continuity state (new episode / the queue was dropped)."""
        self._last_q = None
        self._last_vel = None

    # ---- main entry point -------------------------------------------------
    def retime(
        self,
        chunk: np.ndarray,
        n_out: int,
        state: Optional[np.ndarray] = None,
        anchor: Optional[np.ndarray] = None,
        anchor_vel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Re-time ``chunk`` and return ``(n_out, dof)`` samples spaced by ``dt``.

        ``chunk`` is the full predicted horizon (all of it is used as the path,
        even though only ``n_out`` samples are returned -- the extra waypoints
        are what let the profile brake in time without a terminal slowdown
        inside the part we actually execute). ``state`` is the measured robot
        qpos, used to seed / re-seed the anchor.

        ``anchor`` / ``anchor_vel`` override the carried continuity state for this
        one call. The default (``None``) is the one-chunk-at-a-time path, where
        the last command handed out *is* where the new chunk has to start. Async
        replanning splices into a queue that is still draining, so there the new
        chunk has to start at the point the splice hands over at, which only the
        caller knows (see ``server._splice_seed``).
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != self.dof:
            raise ValueError(
                f"chunk must be [horizon, {self.dof}], got {tuple(chunk.shape)}"
            )
        n_out = max(1, int(n_out))
        info: Dict[str, Any] = {"fallback": None, "anchor": None, "mode": self.mode}

        if anchor is None:
            anchor, info["anchor"] = self._pick_anchor(chunk, state)
            # After `_pick_anchor`, which drops the carried velocity when it has
            # to re-seed from the measured state.
            seed_vel = self._last_vel
        else:
            anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
            if anchor.shape[0] != self.dof:
                raise ValueError(
                    f"anchor must be {self.dof}-dim, got {anchor.shape[0]}"
                )
            info["anchor"] = "caller"
            seed_vel = anchor_vel

        if self.mode == "guard":
            # Steps 2-4 are what re-parameterize time and filter the geometry, and
            # `guard` exists to skip exactly those: the chunk goes to the clamp
            # with its own timing, one waypoint per tick.
            return self._emit(
                np.array(chunk[:n_out], dtype=np.float64), anchor, info, seed_vel
            )

        waypoints = np.vstack([anchor[None, :], chunk])
        n_wp = int(waypoints.shape[0])
        if n_wp < 4:
            # SplineInterpolator needs a handful of points to build a cubic.
            return self._fallback(
                chunk, n_out, anchor, info, f"only {n_wp} waypoints", seed_vel
            )

        waypoints = self._smooth(waypoints)
        if np.abs(np.diff(waypoints[:, self.joint_dims], axis=0)).max() < _MIN_TRAVEL:
            return self._fallback(
                chunk, n_out, anchor, info, "path is stationary", seed_vel
            )

        # Index-based path parameter. Cumulative joint distance would be the
        # other option, but a chunk that dwells produces near-duplicate
        # waypoints and a non-strictly-increasing parameter; the index is always
        # strictly increasing, and the tangent dq/ds still carries the per-step
        # displacement that the limits act on.
        ss = np.arange(n_wp, dtype=np.float64)
        path = _ta.SplineInterpolator(ss, waypoints)

        sd_want = self._initial_path_speed(path, seed_vel)
        traj, sd_start = self._solve(path, sd_want)
        if traj is None and sd_want > 0.0:
            # Last resort: starting from rest always beats handing back a chunk
            # that was never re-timed (it costs a brake/accelerate at the seam).
            traj, sd_start = self._solve(path, 0.0)
            if traj is not None:
                info["note"] = f"entry speed {sd_want:.2f} infeasible, restarted from rest"
        info["sd_start"] = float(sd_start)
        info["sd_wanted"] = float(sd_want)
        if traj is None:
            return self._fallback(
                chunk, n_out, anchor, info, "toppra found no trajectory", seed_vel
            )
        duration = float(getattr(traj, "duration", 0.0) or 0.0)
        if not np.isfinite(duration) or duration <= 0.0:
            return self._fallback(
                chunk, n_out, anchor, info, f"bad duration {duration}", seed_vel
            )

        ts = np.clip(np.arange(1, n_out + 1, dtype=np.float64) * self.dt, 0.0, duration)
        out = np.asarray(traj(ts), dtype=np.float64).reshape(len(ts), self.dof)
        vel = np.asarray(traj(ts, 1), dtype=np.float64).reshape(len(ts), self.dof)

        # The spline rings around a gripper step; clamp it back into the range
        # the chunk actually asked for so a cubic overshoot cannot command the
        # gripper past its physical travel.
        for i in self.gripper_dims:
            lo, hi = waypoints[:, i].min(), waypoints[:, i].max()
            np.clip(out[:, i], lo, hi, out=out[:, i])

        nominal = (n_wp - 1) * self.dt
        info["duration"] = duration
        info["time_scale"] = (duration / nominal) if nominal > 0 else None
        info["consumed"] = float(min(float(ts[-1]), duration) / duration)
        info["peak_vel"] = float(
            np.max(np.abs(vel[:, self.joint_dims]) / self._vlim[self.joint_dims, 1])
        )

        return self._emit(out, anchor, info, seed_vel)

    def guard(
        self,
        samples: np.ndarray,
        anchor: np.ndarray,
        anchor_vel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clamp an already-assembled command stream. Stateless, on purpose.

        ``retime`` owns the anchor/velocity bookkeeping for the path where one
        chunk is queued at a time, because there the chunk it was handed *is* the
        command stream. Async replanning breaks that: the stream that reaches the
        arm is a splice of a chunk still being handed out and a fresher one (see
        ``server._merge_chunk``), and neither of them is what the limits have
        to hold on. So the caller assembles the stream, tracks what it actually
        emitted, and re-checks the result here -- touching none of the continuity
        state, which by then describes the tail of a chunk the arm never reached.

        Returns the clamped stream plus ``guard_bite`` / ``guard_vel``: the worst
        pre-clamp acceleration and velocity as a fraction of the limit, i.e. how
        much of this stream was not admissible to begin with.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != self.dof:
            raise ValueError(
                f"samples must be [n, {self.dof}], got {tuple(samples.shape)}"
            )
        anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
        if anchor.shape[0] != self.dof:
            raise ValueError(f"anchor must be {self.dof}-dim, got {anchor.shape[0]}")
        out, _vel, worst_acc, worst_vel = self._enforce_discrete_limits(
            samples, anchor, anchor_vel
        )
        return out.astype(np.float32), {
            "guard_bite": worst_acc, "guard_vel": worst_vel,
        }

    # ---- internals --------------------------------------------------------
    def _pick_anchor(
        self, chunk: np.ndarray, state: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, str]:
        """Where the re-timed path starts (see step 1 of the module docstring)."""
        measured = None
        if state is not None:
            s = np.asarray(state, dtype=np.float64).reshape(-1)
            if s.shape[0] == self.dof:
                measured = s
            else:
                logger.warning(
                    "TOPP: ignoring a %d-dim state for a %d-dim action.",
                    s.shape[0], self.dof,
                )

        if self._last_q is None:
            if measured is not None:
                return measured, "measured state"
            return chunk[0].copy(), "chunk[0]"

        if measured is not None and self.max_anchor_error > 0.0:
            err = float(
                np.abs(self._last_q[self.joint_dims] - measured[self.joint_dims]).max()
            )
            if err > self.max_anchor_error:
                logger.warning(
                    "TOPP: last command is %.3f rad from the measured state "
                    "(> %.3f); the arm is not tracking, re-seeding the path from "
                    "the measured state.",
                    err, self.max_anchor_error,
                )
                self._last_vel = None  # its timing no longer describes the arm
                return measured, "measured state (re-seeded)"

        return self._last_q.copy(), "last command"

    def _smooth(self, waypoints: np.ndarray) -> np.ndarray:
        """Savitzky-Golay filter the joint columns, pinning the anchor row."""
        n = int(waypoints.shape[0])
        win = min(self.smooth_window, n if n % 2 == 1 else n - 1)
        if win % 2 == 0:
            win -= 1
        if win < 3 or self.smooth_polyorder < 1 or win <= self.smooth_polyorder:
            return waypoints
        out = waypoints.copy()
        out[:, self.joint_dims] = _savgol(
            waypoints[:, self.joint_dims], win, self.smooth_polyorder, axis=0
        )
        out[0] = waypoints[0]  # the seam with the previous chunk must not move
        return out

    def _initial_path_speed(self, path: Any, vel: Optional[np.ndarray]) -> float:
        """Previous exit velocity, projected onto the new path's tangent at s=0.

        The two chunks are different paths, so the path speed ``sd`` itself does
        not carry over -- the joint velocity does. Least-squares projecting it
        onto ``dq/ds`` keeps the component that still points along the new path
        and drops the rest, which is what lets consecutive chunks join without
        braking to a stop at every replan. A negative projection means the new
        path heads back the way we came, and there is nothing to carry.

        Feasibility is not checked here -- ``_solve`` clamps against the exact
        controllable set, which subsumes any bound derivable from the tangent.
        """
        if vel is None:
            return 0.0
        vel = np.asarray(vel, dtype=np.float64).reshape(-1)
        tangent = np.asarray(path(0.0, 1), dtype=np.float64).reshape(-1)
        j = self.joint_dims
        denom = float(tangent[j] @ tangent[j])
        if denom < 1e-12:
            return 0.0
        sd = float(vel[j] @ tangent[j]) / denom
        return float(sd) if np.isfinite(sd) and sd > 0.0 else 0.0

    def _solve(self, path: Any, sd_start: float) -> Tuple[Optional[Any], float]:
        """Parameterize ``path``, entering at (at most) ``sd_start``.

        Returns ``(trajectory, entry speed actually used)``.

        The requested entry speed is clamped to toppra's *controllable set* at
        s=0 rather than to a velocity-derived bound. Being under the velocity
        limit is necessary but not sufficient: the profile must also be able to
        brake through whatever the path does later, so the truly admissible entry
        speed is often well below it. Asking for more than that makes the problem
        infeasible, toppra returns None, and we would fall back to entering at
        rest -- a brake-and-re-accelerate at every replan boundary, which is the
        stop-and-go this module exists to remove. Clamping instead keeps the
        fastest entry the path can actually support.
        """
        try:
            instance = _ta_algo.TOPPRA(
                [
                    _ta_constraint.JointVelocityConstraint(self._vlim),
                    _ta_constraint.JointAccelerationConstraint(self._alim),
                ],
                path,
                solver_wrapper="seidel",
            )
            sd_start = float(sd_start)
            if sd_start > 0.0:
                # x = sd^2; [0] is the gridpoint at s=0. Stay just inside the
                # boundary, which is itself only admissible up to rounding.
                controllable = np.asarray(instance.compute_controllable_sets(0.0, 0.0))
                sd_max = float(np.sqrt(max(float(controllable[0, 1]), 0.0)))
                sd_start = min(sd_start, 0.999 * sd_max)
            return instance.compute_trajectory(sd_start, 0.0), sd_start
        except Exception as exc:  # noqa: BLE001 - never break a rollout on a solve
            logger.warning("TOPP: solver raised %s", exc)
            return None, 0.0

    def _emit(
        self,
        samples: np.ndarray,
        anchor: np.ndarray,
        info: Dict[str, Any],
        vel: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the discrete guard, record the continuity state, hand back samples."""
        out, cmd_vel, worst_acc, worst_vel = self._enforce_discrete_limits(
            samples, anchor, vel
        )
        info["guard_bite"] = worst_acc
        info["guard_vel"] = worst_vel
        self._last_q = out[-1].copy()
        self._last_vel = cmd_vel.copy()
        return out.astype(np.float32), info

    def _enforce_discrete_limits(
        self,
        samples: np.ndarray,
        anchor: np.ndarray,
        vel: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Clamp the emitted stream so its finite differences respect the limits.

        A first-order tracker over the samples, starting at ``anchor`` with
        velocity ``vel`` (the one the command stream was last emitted at; ``None``
        means from rest): each step takes the velocity the sample asks for, limits
        how much it may differ from the previous velocity (acceleration) and how
        large it may be (velocity), then integrates. By construction
        ``|dq/dt| <= max_vel`` and ``|dv/dt| <= max_acc`` on the result, whatever
        was handed in.

        This is the same shape as the replay loop's tracker in the robot
        control repo, one control period instead of 1 ms, and it exists because
        the client's ``send_action`` goes straight to ``set_qpos``, so the
        command stream *is* what the hardware executes.

        Gripper dimensions are copied through untouched -- they are not joints,
        and delaying an open/close costs a grasp.

        Returns ``(clamped samples, final joint velocity, worst pre-clamp
        acceleration, worst pre-clamp velocity)``, the last two as a fraction of
        their limit -- how much of what was handed in was inadmissible, which is
        the number to look at when deciding whether the limits are honest.
        """
        out = np.array(samples, dtype=np.float64, copy=True)
        j = self.joint_dims
        vmax = self._vlim[j, 1]
        amax = self._alim[j, 1]
        dv_max = amax * self.dt

        q = np.asarray(anchor, dtype=np.float64)[j].copy()
        v = (
            np.zeros(j.size)
            if vel is None
            else np.asarray(vel, dtype=np.float64).reshape(-1)[j].copy()
        )
        worst_acc = 0.0
        worst_vel = 0.0
        for k in range(out.shape[0]):
            v_want = (out[k, j] - q) / self.dt
            worst_acc = max(worst_acc, float(np.max(np.abs(v_want - v) / dv_max)))
            worst_vel = max(worst_vel, float(np.max(np.abs(v_want) / vmax)))
            v = np.clip(np.clip(v_want, v - dv_max, v + dv_max), -vmax, vmax)
            q = q + v * self.dt
            out[k, j] = q
        # Full-width velocity: the gripper entries are unused (the next chunk only
        # projects the joint dims) but keep the vector rectangular.
        cmd_vel = np.zeros(self.dof)
        cmd_vel[j] = v
        return out, cmd_vel, worst_acc, worst_vel

    def _fallback(
        self,
        chunk: np.ndarray,
        n_out: int,
        anchor: np.ndarray,
        info: Dict[str, Any],
        reason: str,
        vel: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Un-re-timed chunk, still bounded by the discrete guard. Say why."""
        info["fallback"] = reason
        return self._emit(np.array(chunk[:n_out], dtype=np.float64), anchor, info, vel)
