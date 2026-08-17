#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict, replace

import capnp
from openpilot.cereal import messaging, log, custom
from opendbc.car.structs import car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame


# Spatial/kinematic association (the LeadTrackingSpatial toggle). The Rivian Mando radar renumbers
# track ids ~60/sec (median track life ~1 frame; corpus drive 00000231/24-25), so id-keyed hysteresis
# is INERT on it — it has no stable id to stay sticky to (measured: Tier-2 == Tier-1 exactly on that
# drive). Instead we bias the matcher toward the radar track nearest the PREVIOUS emitted lead in
# position AND relative velocity — id-free, so it survives the renumbering. Velocity-agreement is the
# key discriminator: it rejects roadside clutter and the far/near mis-grabs that share the lead's
# lane but not its speed. User-selectable on any brand (default off, id-keyed); validated on Rivian.
# Tuned ("vel_cont L5/Lv1") on the id-churn corpus: vs Tier-2 id-hysteresis, -17% brake-jumps alone
# and (stacked under the Tier-3 coast) -17% vs the shipped Tier-3, 0 radar-corroborated streak regr.
SPATIAL_POS_RC = 5.0   # m   — position-continuity length scale
SPATIAL_VEL_RC = 1.0   # m/s — relative-velocity-continuity length scale

# Tier 3 "vision-coast".
#
# A phantom is a radar lead whose speed disagrees with the (confident)
# vision lead AND that arrived via an abrupt radar discontinuity the camera does NOT corroborate, either:
#   - the chosen track-id just changed, or
#   - the radar lead jumped laterally while the vision lead held still
#
# On such a frame, hold the last-good lead for up to COAST_MAX_S instead of
# emitting the phantom; then hand back to the radar until a real lead returns.
#
# The corroborating discontinuity is what lets the speed gate stay loose without coasting on clean frames.
# Holding the last-good (Kalman-smoothed) radar lead beats switching to the noisier vision lead.
COAST_V_GROSS = 5.0          # m/s — speed disagreement vs vision
                             #       a corroborating discontinuity (another sus event)
                             #       is also required, so this stays strict
COAST_LAT_RADAR_JUMP = 1.0   # m   — chosen-lead yRel jump
COAST_LAT_VISION_STILL = 0.5 # m   — vision lead barely moved laterally (so the jump is radar-only)
COAST_MAX_S = 0.3            # s   — hard cap on continuous coasting, then hand back to the radar.
                             #       bounds the worst-case coast distance to v·0.3
                             #       assuming freeway speeds:
                             #       90 mph (~41 m/s) (faster than most cruise)
                             #       that comes out to: 12.3 m coasting
                             #       this does not include the COAST_NEAR_DREL lockout
                             #       so any close range radar lead would immediately cancel the coast
COAST_NEAR_DREL = 40.0       # m   — close-cut-in failsafe: never coast past a radar lead nearer than
                             #       this. A close return is a hazard — respond to it, don't hold a
                             #       stale far lead. The inverse of the gross-distance trigger: here
                             #       closeness SUPPRESSES coast. Set from the corpus coast-distance
                             #       distribution: coasts under ~40 m are ~100% vision-confirmed real
                             #       objects (often stopped, stop-and-go), where coasting past = a
                             #       rear-end risk; the phantom benefit lives >60 m. 0 disables.
COAST_VIS_PROB = 0.5         #     — only trust the phantom judgment when the vision lead is confident


@dataclass
class RadarLead:
  """A radarState leadOne/leadTwo estimate. Field names mirror cereal LeadData, so an instance
  maps onto the cereal struct via asdict(); the defaults match cereal's defaults, so a bare
  RadarLead() is the 'no lead' value (present=False) — it replaces the old {'present': False} dict."""
  present: bool = False
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  vLead: float = 0.0
  vLeadK: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 0.0
  modelProb: float = 0.0
  radar: bool = False
  radarTrackId: int = -1


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0) -> RadarLead:
    return RadarLead(
      dRel=float(self.dRel),
      yRel=float(self.yRel),
      vRel=float(self.vRel),
      vLead=float(self.vLead),
      vLeadK=float(self.vLeadK),
      aLeadK=float(self.aLeadK),
      aLeadTau=float(self.aLeadTau.x),
      present=True,
      modelProb=model_prob,
      radar=True,
      radarTrackId=self.identifier,
    )

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track],
                          prev_track_id: int | None = None, stickiness: float = 5.0,
                          prev_lead_state: tuple[float, float, float] | None = None,
                          pos_rc: float = SPATIAL_POS_RC, vel_rc: float = SPATIAL_VEL_RC) -> Track | None:
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    base = prob_d * prob_y * prob_v
    # Hysteresis: bias toward the lead we were already tracking so a challenger must beat it by a
    # margin before we switch — prevents frame-to-frame flipping when two radar tracks have similar
    # probability.
    if prev_lead_state is not None:
      # Spatial/kinematic (Rivian): bias the track nearest the previous emitted lead in position AND
      # relative velocity. Id-free, so it survives the Rivian radar's track-id renumbering (~60/sec)
      # that makes id-keyed stickiness inert. Velocity-agreement rejects clutter and far/near
      # mis-grabs that share the lead's lane but not its speed. See SPATIAL_POS_RC / SPATIAL_VEL_RC.
      prev_dRel, prev_yRel, prev_vRel = prev_lead_state
      pos = math.hypot(c.dRel - prev_dRel, c.yRel - prev_yRel)
      base *= math.exp(-pos / pos_rc) * math.exp(-abs(c.vRel - prev_vRel) / vel_rc)
    elif prev_track_id is not None and c.identifier == prev_track_id:
      # Id-keyed (brands with stable radar ids): bias the previously-chosen track, scaled by the
      # lateral-agreement probability so the held track releases its bonus when it drifts away from
      # the camera's predicted lateral position (only stay sticky while the camera still supports it).
      base *= 1.0 + (stickiness - 1.0) * prob_y
    return base

  track = max(tracks.values(), key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float, lead_prob: float) -> RadarLead:
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return RadarLead(
    dRel=float(lead_msg.x[0] - RADAR_TO_CAMERA),
    yRel=float(-lead_msg.y[0]),
    vRel=float(lead_v_rel_pred),
    vLead=float(v_ego + lead_v_rel_pred),
    vLeadK=float(v_ego + lead_v_rel_pred),
    aLeadK=float(lead_msg.a[0]),
    aLeadTau=0.3,
    modelProb=float(lead_prob),
    present=True,
    radar=False,
    radarTrackId=-1,
  )


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, lead_prob: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
             low_speed_override: bool = True, prev_track_id: int | None = None,
             prev_lead_state: tuple[float, float, float] | None = None,
             pos_rc: float = SPATIAL_POS_RC, vel_rc: float = SPATIAL_VEL_RC) -> RadarLead:
  # Determine leads, this is where the essential logic happens
  if len(tracks) > 0 and ready and lead_prob > .5:
    track = match_vision_to_track(v_ego, lead_msg, tracks, prev_track_id=prev_track_id,
                                  prev_lead_state=prev_lead_state, pos_rc=pos_rc, vel_rc=vel_rc)
  else:
    track = None

  lead = RadarLead()
  if track is not None:
    lead = track.get_RadarState(lead_prob)
    lead = get_custom_yrel(CP, CP_SP, lead, lead_msg)
  elif (track is None) and ready and (lead_prob > .5):
    lead = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego, lead_prob)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead.present) or (closest_track.dRel < lead.dRel):
        lead = closest_track.get_RadarState()

  return lead


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead: RadarLead,
                    lead_msg: capnp._DynamicStructReader) -> RadarLead:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead.yRel = float(-lead_msg.y[0])

  return lead


class RadarD:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0
    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)
    self.lead_prob_filters = [FirstOrderFilter(0.0, 0.2, DT_MDL) for _ in range(2)]

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

    # Hysteresis state for match_vision_to_track. With the LeadTrackingSpatial toggle we anchor on
    # the previous emitted lead's (dRel, yRel, vRel) — id-free spatial/kinematic continuity (see
    # SPATIAL_POS_RC), built for radars that renumber track ids (e.g. Rivian) but selectable on any
    # brand. Toggle off (default): id-keyed stickiness on the last selected radar trackId (also
    # reused by the Tier-3 id-changed corroborator). Refreshed ~1 Hz alongside the tier below;
    # offline harnesses set use_spatial_assoc directly.
    self.use_spatial_assoc = False
    self.prev_lead_track_id: dict[int, int | None] = {0: None, 1: None}
    self.prev_lead_state: dict[int, tuple[float, float, float] | None] = {0: None, 1: None}

    # behavior tier: 1=no hysteresis (greedy match), 2=hysteresis (spatial/kinematic with the
    # LeadTrackingSpatial toggle, id-keyed otherwise), 3=hysteresis + coast (hold last-good through phantoms AND
    # radar<->vision handoff drops). Chosen by the LeadTrackingMode UI selector, which stores a button index
    # (0/1/2 -> tier 1/2/3); main() refreshes it ~1 Hz via the DEC throttled-read pattern (never
    # per-frame disk I/O). update() never reads Params, so offline eval harnesses just set
    # self.lead_tracking_mode directly and it sticks. Defaults to Tier 2 (the validated behavior).
    self.params = Params()
    self.frame = 0
    self.lead_tracking_mode = 2
    # Tier 3 vision-coast state (per lead): last lead we trusted, how long we've coasted, and
    # the previous-frame lateral positions powering the radar-lateral-jump corroborator.
    self.last_good_lead: dict[int, RadarLead | None] = {0: None, 1: None}
    self.coast_frames: dict[int, int] = {0: 0, 1: 0}
    # per-lead coast flag, published as radarStateSP for the onroad UI indicator (chevron tint
    # + developer-UI element). Reset each update(); set by _vision_coast when a lead is held.
    self.lead_coasting: dict[int, bool] = {0: False, 1: False}
    self.prev_lead_yRel: dict[int, float | None] = {0: None, 1: None}
    self.prev_vision_y: dict[int, float | None] = {0: None, 1: None}
    # Tier 3 thresholds — instance attrs so they can be swept at runtime during tuning
    self.coast_v_gross = COAST_V_GROSS
    self.coast_lat_radar_jump = COAST_LAT_RADAR_JUMP
    self.coast_lat_vision_still = COAST_LAT_VISION_STILL
    self.coast_max_s = COAST_MAX_S
    self.coast_vis_prob = COAST_VIS_PROB
    self.coast_near_dRel = COAST_NEAR_DREL
    # spatial-association length scales (Rivian) — instance attrs so they can be swept at runtime
    self.spatial_pos_rc = SPATIAL_POS_RC
    self.spatial_vel_rc = SPATIAL_VEL_RC

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    # clear the per-lead coast flags for this frame (set again below only if Tier 3 holds a lead)
    self.frame += 1
    self.lead_coasting[0] = self.lead_coasting[1] = False

    self.ready = sm.seen['modelV2']

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead)

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      for i in range(2):
        # Asymmetric filter on lead prob to keep lead when uncertain
        lead_prob = leads_v3[i].prob
        if lead_prob > self.lead_prob_filters[i].x:
          self.lead_prob_filters[i].x = lead_prob
        else:
          self.lead_prob_filters[i].update(lead_prob)

      tier = self.lead_tracking_mode
      # Tier 1: no hysteresis (greedy match). Tier 2/3: hysteresis — spatial/kinematic anchor with
      # the LeadTrackingSpatial toggle (radars that renumber ids), id-keyed otherwise.
      sticky = [None, None]
      anchor: list[tuple[float, float, float] | None] = [None, None]
      if tier >= 2:
        if self.use_spatial_assoc:
          anchor = [self.prev_lead_state[0], self.prev_lead_state[1]]
        else:
          sticky = [self.prev_lead_track_id[0], self.prev_lead_track_id[1]]
      one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, self.lead_prob_filters[0].x,
                     self.CP, self.CP_SP, low_speed_override=True, prev_track_id=sticky[0], prev_lead_state=anchor[0],
                     pos_rc=self.spatial_pos_rc, vel_rc=self.spatial_vel_rc)
      two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, self.lead_prob_filters[1].x,
                     self.CP, self.CP_SP, low_speed_override=False, prev_track_id=sticky[1], prev_lead_state=anchor[1],
                     pos_rc=self.spatial_pos_rc, vel_rc=self.spatial_vel_rc)
      # The spatial anchor tracks the matcher's OWN pick (pre-coast), so a long coast hold doesn't
      # freeze the anchor onto a stale spot — capture the matched leads before the coast may replace them.
      precoast = (one, two)
      # Tier 3: vision-coast — hold the last-good lead through a gross phantom (e.g. a
      # single-frame radar dropout) instead of emitting it.
      if tier >= 3:
        one, c0 = self._vision_coast(0, one, leads_v3[0])
        two, c1 = self._vision_coast(1, two, leads_v3[1])
        self.lead_coasting[0] = c0
        self.lead_coasting[1] = c1
      else:
        self.coast_frames[0] = self.coast_frames[1] = 0
      self.radar_state.leadOne = asdict(one)
      self.radar_state.leadTwo = asdict(two)
      # remember the chosen radar track id (hysteresis bonus + id-changed corroborator) and the
      # lateral positions (radar-lateral-jump corroborator) for the next frame
      for i, ld in ((0, one), (1, two)):
        if ld.present and ld.radar and ld.radarTrackId >= 0:
          self.prev_lead_track_id[i] = ld.radarTrackId
          self.prev_lead_yRel[i] = ld.yRel
        else:
          self.prev_lead_track_id[i] = None
          self.prev_lead_yRel[i] = None
        # spatial-association anchor: the matcher's own PRE-coast pick (radar OR vision fallback),
        # id-free. Follows a closing lead down frame-by-frame; cleared when there's no lead to anchor.
        pc = precoast[i]
        self.prev_lead_state[i] = (pc.dRel, pc.yRel, pc.vRel) if pc.present else None
        self.prev_vision_y[i] = float(leads_v3[i].y[0]) if len(leads_v3[i].y) else None

  def _vision_coast(self, i: int, lead: RadarLead,
                    vis: capnp._DynamicStructReader) -> tuple[RadarLead, bool]:
    """Tier 3 coast. Hold the last-good radar lead instead of emitting the current one when this
    frame is untrustworthy — either a phantom radar lead (is_phantom) OR a radar->vision handoff we
    can smooth (_is_handoff_drop) — bounded by COAST_MAX_S, after which we hand back to the radar
    until a real lead returns. Returns (possibly-held lead, coasting?). Holding the last-good (Kalman-
    smoothed) radar lead beats emitting either the phantom or the noisier raw vision fallback."""
    hold = self.is_phantom(i, lead, vis) or self._is_handoff_drop(i, lead)

    # When the frame is trustworthy, reset the coast timer and refresh last-good to this frame's lead
    # — or clear it to None when there's no radar lead. Never keep a stale last-good around: a
    # trustworthy lead is safe to hold, and if there's nothing to hold we'd rather not coast at all
    # than coast onto a stale value.
    if not hold:
      self.coast_frames[i] = 0
      self.last_good_lead[i] = replace(lead) if (lead.present and lead.radar) else None
      return lead, False

    # --- BELOW HERE WE DO NOT TRUST THE CURRENT LEAD (phantom radar, or a radar->vision handoff) ---

    # We only want to coast for a max time, so that becomes a max number of frames; if we hit that
    # limit we drop back to returning the current (untrusted) lead, which is safer than coasting on.
    cap = int(round(self.coast_max_s / DT_MDL))

    # if we have a last-good lead to coast with and are within the coast budget, coast
    if self.last_good_lead[i] is not None and self.coast_frames[i] < cap:
      self.coast_frames[i] += 1
      return replace(self.last_good_lead[i]), True

    # no last-good to coast with, or we hit the coast limit → drop back to the current (untrusted) lead
    return lead, False

  def _is_handoff_drop(self, i: int, lead: RadarLead) -> bool:
    """Tier 3 handoff smoothing. A radar->vision handoff: match_vision_to_track rejected the radar
    track — overwhelmingly on a distance disagreement (corpus: 97.5% of r2v drops) — so the emitted
    lead fell to the noisier raw vision fallback, a discontinuity. When we still hold a recent radar
    lead that is the SAME object the camera still sees, coast it across the excursion (same hold
    mechanism + cap as the phantom case) instead of emitting the jump; bridging the drop also avoids
    the paired jump when the radar later reacquires.

    Safety — only smooth a genuine range hiccup, never a wrong/changed lead:
      - SPEED must agree (held vs the camera's current lead). Radar relative-speed is reliable, so a
        gross speed disagreement means a DIFFERENT vehicle (a wrong/stale radar track) or a real
        decel the radar just lost — in both cases follow vision, never hold a stale lead. This is
        what keeps held frames from extending a 'stuck-on-wrong-vehicle' gross streak.
      - DISTANCE may differ — radar range is more accurate than mono-camera range, so a pure range
        hiccup (speed agrees, distance jumps) is exactly what we DO smooth — EXCEPT when vision
        reports a genuinely nearer lead (a cut-in it could be right about) or a near hazard."""
    # only a present vision-derived lead, with a radar lead in hand to fall back on, is a handoff
    if not (lead.present and not lead.radar) or self.last_good_lead[i] is None:
      return False
    held = self.last_good_lead[i]
    # close-cut-in failsafe (shared with is_phantom): a near lead is a hazard to respond to, not smooth
    if 0.0 < lead.dRel < self.coast_near_dRel:
      return False
    # same object? a gross speed disagreement = a different/changed lead → follow vision, don't hold
    if abs(held.vLead - lead.vLead) > self.coast_v_gross:
      return False
    # vision sees the lead meaningfully nearer than the one we're holding → a real approach, obey it.
    # Same 25%/5m tolerance as match_vision_to_track's dist_sane gate, applied to the held distance.
    if lead.dRel < held.dRel - max(held.dRel * 0.25, 5.0):
      return False
    return True

  def is_phantom(self, i: int, lead: RadarLead, vis: capnp._DynamicStructReader) -> bool:
    """Tier 3 per-frame verdict: do we think this radar lead is a phantom we should NOT trust? It
    is, when its speed grossly disagrees with the confident vision lead AND it arrived via an abrupt
    radar discontinuity the camera does not corroborate (the chosen track-id changed, or the radar
    lead jumped laterally while the vision lead held still). Pure predicate — no coast cap, no state
    mutation — so eval harnesses can label frames with it directly. Cheap checks short-circuit first."""
    if not (lead.present and lead.radar):
      return False
    # Close-cut-in failsafe: a very close radar return is a real hazard (e.g. a side cut-in), never
    # a phantom to coast past — respond to it instead of holding a stale far lead. The inverse of a
    # distance trigger: closeness vetoes the phantom verdict outright.
    if 0.0 < lead.dRel < self.coast_near_dRel:
      return False
    # A phantom's speed grossly disagrees with the vision lead; if it agrees (or we can't tell),
    # it isn't a phantom.
    if self._speed_agrees_with_vision(lead, vis):
      return False

    # --- BELOW HERE WE SUSPECT SOMETHING FISHY ---
    # but we need proof to back that up. So far we know:
    #   - this is a medium-distance object (the close-cut-in failsafe above already let it through)
    #   - it's moving at a dramatically different speed from the vision lead
    # That alone is suspicious but not damning, so we look for a radar discontinuity to corroborate:
    #   - did the radar swap to a new track id?
    #   - did the radar lead hop dramatically to the side (while vision didn't)?
    # Either can mean the radar locked onto something like a stationary object in an adjacent lane —
    # e.g. driving past a row of stopped cars, or reflectors between lanes (toll plaza). There ARE
    # real situations like this where you'd want to brake, which is why we keep the distance gate
    # (nothing fires too close — 40 m default, giving vision time to pick it up) and the vision check.

    # Corroborator 1: the chosen radar track-id just changed (a dropout grabbed a different object).
    if lead.radarTrackId != -1 and lead.radarTrackId != self.prev_lead_track_id[i]:
      return True

    # Corroborator 2: the radar lead jumped laterally while the vision lead held still.
    prev_lead_yRel, prev_vision_y = self.prev_lead_yRel[i], self.prev_vision_y[i]
    if prev_lead_yRel is None or prev_vision_y is None:
      return False

    has_dramatic_lateral_radar_jump = abs(lead.yRel - prev_lead_yRel) > self.coast_lat_radar_jump
    vision_lead_laterally_stable = abs(vis.y[0] - prev_vision_y) < self.coast_lat_vision_still
    return has_dramatic_lateral_radar_jump and vision_lead_laterally_stable

  def _speed_agrees_with_vision(self, lead: RadarLead, vis: capnp._DynamicStructReader) -> bool:
    """True when the radar lead's speed is consistent with the vision lead — i.e. NOT a gross
    disagreement. Also True when there's no basis to call a disagreement (ego ~stopped, or no
    confident vision lead), so those frames default to 'not a phantom'. The negation is the phantom
    speed symptom: a dropout that re-points the radar slot at a slower/closer object craters the
    speed relative to the lead the camera still tracks."""
    if self.v_ego <= V_EGO_STATIONARY:
      return True
    if not len(vis.x) or not len(vis.y) or vis.prob <= self.coast_vis_prob:
      return True
    return abs(lead.vLead - vis.v[0]) <= self.coast_v_gross

  def _read_lead_tracking_params(self) -> None:
    # The UI selector stores a button index (0/1/2); map it to the tier (1/2/3). The spatial toggle
    # maps straight onto use_spatial_assoc. Called ~1 Hz from main() only (NOT update()), so a
    # harness that sets lead_tracking_mode / use_spatial_assoc directly isn't clobbered. Tolerate
    # absent keys (e.g. before a params rebuild) by keeping the current values.
    try:
      idx = self.params.get("LeadTrackingMode", return_default=True)
      if idx is not None:
        self.lead_tracking_mode = int(idx) + 1
      self.use_spatial_assoc = self.params.get_bool("LeadTrackingSpatial")
    except Exception:
      pass

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)

    # sunnypilot: per-lead Tier-3 vision-coast state for the onroad UI indicator
    radar_sp = messaging.new_message("radarStateSP")
    radar_sp.valid = self.radar_state_valid
    radar_sp.radarStateSP.leadOneCoasting = self.lead_coasting[0]
    radar_sp.radarStateSP.leadTwoCoasting = self.lead_coasting[1]
    pm.send("radarStateSP", radar_sp)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(Params().get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'radarTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState', 'radarStateSP'])

  RD = RadarD(CP, CP_SP, CP.radarDelay)

  while 1:
    sm.update()

    # refresh the lead-tracking params (tier selector + spatial toggle) ~1 Hz — the same sm.frame-
    # gated main-loop param-read idiom as selfdrived/card/paramsd (cheap; never per-frame disk I/O).
    # sm.frame is 0 on the first pass so the settings are live from boot. Kept out of update() so
    # offline harnesses pin the behavior by setting the attrs directly.
    if sm.frame % int(1. / DT_MDL) == 0:
      RD._read_lead_tracking_params()
    RD.update(sm, sm['radarTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
