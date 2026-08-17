#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict

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
# Tuned ("vel_cont L5/Lv1") on the id-churn corpus: vs id-keyed hysteresis, -17% brake-jumps,
# 0 radar-corroborated streak regressions.
SPATIAL_POS_RC = 5.0   # m   — position-continuity length scale
SPATIAL_VEL_RC = 1.0   # m/s — relative-velocity-continuity length scale


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
    # brand. Toggle off (default): id-keyed stickiness on the last selected radar trackId.
    # Refreshed ~1 Hz alongside the tier below; offline harnesses set use_spatial_assoc directly.
    self.use_spatial_assoc = False
    self.prev_lead_track_id: dict[int, int | None] = {0: None, 1: None}
    self.prev_lead_state: dict[int, tuple[float, float, float] | None] = {0: None, 1: None}

    # behavior tier: 1=no hysteresis (greedy match), 2=hysteresis (spatial/kinematic with the
    # LeadTrackingSpatial toggle, id-keyed otherwise). Chosen by the LeadTrackingMode UI selector, which stores a button index
    # (0/1 -> tier 1/2); main() refreshes it ~1 Hz via the DEC throttled-read pattern (never
    # per-frame disk I/O). update() never reads Params, so offline eval harnesses just set
    # self.lead_tracking_mode directly and it sticks. Defaults to Tier 2 (the validated behavior).
    self.params = Params()
    self.frame = 0
    self.lead_tracking_mode = 2
    # spatial-association length scales (Rivian) — instance attrs so they can be swept at runtime
    self.spatial_pos_rc = SPATIAL_POS_RC
    self.spatial_vel_rc = SPATIAL_VEL_RC

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.frame += 1

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
      # Tier 1: no hysteresis (greedy match). Tier 2: hysteresis — spatial/kinematic anchor with
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
      self.radar_state.leadOne = asdict(one)
      self.radar_state.leadTwo = asdict(two)
      # remember the chosen radar track id (id-keyed hysteresis bonus) and the matcher's pick
      # (spatial anchor) for the next frame
      for i, ld in ((0, one), (1, two)):
        if ld.present and ld.radar and ld.radarTrackId >= 0:
          self.prev_lead_track_id[i] = ld.radarTrackId
        else:
          self.prev_lead_track_id[i] = None
        # spatial-association anchor: the matcher's own pick (radar OR vision fallback), id-free.
        # Follows a closing lead down frame-by-frame; cleared when there's no lead to anchor.
        self.prev_lead_state[i] = (ld.dRel, ld.yRel, ld.vRel) if ld.present else None

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
  pm = messaging.PubMaster(['radarState'])

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
