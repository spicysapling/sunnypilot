"""Unit tests for radard lead selection — the RadarLead type, vision↔radar matching, the tier-2
hysteresis (id-keyed by default, spatial/kinematic with the LeadTrackingSpatial toggle), and the
tier-3 vision-coast. These drive the pure-Python functions/objects directly (no process_replay / msgq),
so they run on macOS as well as in CI."""
from dataclasses import asdict

from openpilot.cereal import messaging, log, custom
from opendbc.car.structs import car
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.radard import (
  RadarLead, Track, KalmanParams, RadarD, get_lead, match_vision_to_track, RADAR_TO_CAMERA,
  SPATIAL_POS_RC, SPATIAL_VEL_RC,
)

V_EGO = 25.0


def vision_msg(x, y, v, prob=1.0, x_std=2.0, y_std=1.0, v_std=1.0, a=0.0):
  """Build a modelV2 message with one lead. Return the message (keep it in scope) — index
  `.modelV2.leadsV3[0]` to get the reader the radard functions consume."""
  msg = messaging.new_message('modelV2')
  msg.modelV2.leadsV3 = [{'prob': prob, 'x': [x], 'xStd': [x_std], 'y': [y], 'yStd': [y_std],
                          'v': [v], 'vStd': [v_std], 'a': [a]}]
  return msg


def make_track(identifier, d_rel, y_rel, v_rel, v_ego=V_EGO):
  t = Track(identifier, v_rel + v_ego, KalmanParams(DT_MDL))
  t.update(d_rel, y_rel, v_rel, v_rel + v_ego)
  return t


def make_radard():
  return RadarD(car.CarParams.new_message(), custom.CarParamsSP.new_message(), delay=0.0)


class TestRadarLead(OpenpilotTestCase):
  def test_default_is_no_lead(self):
    lead = RadarLead()
    assert lead.present is False and lead.radar is False and lead.radarTrackId == -1

  def test_cereal_roundtrip(self):
    # a full radar lead maps onto the cereal LeadData via asdict() with no loss
    lead = RadarLead(present=True, dRel=52.3, yRel=-1.2, vRel=-3.0, vLead=22.0, vLeadK=22.1,
                     aLeadK=0.4, aLeadTau=1.5, modelProb=0.9, radar=True, radarTrackId=7)
    rs = log.RadarState.new_message()
    rs.leadOne = asdict(lead)
    # cereal LeadData floats are Float32, so compare floats with tolerance and the rest exactly
    for f in ("present", "radar", "radarTrackId"):
      assert getattr(rs.leadOne, f) == getattr(lead, f), f
    for f in ("dRel", "yRel", "vRel", "vLead", "vLeadK", "aLeadK", "aLeadTau", "modelProb"):
      assert abs(getattr(rs.leadOne, f) - getattr(lead, f)) < 1e-4, f

  def test_no_lead_cereal_matches_defaults(self):
    # the bare RadarLead() must produce the same struct the old {'present': False} dict did
    rs = log.RadarState.new_message()
    rs.leadTwo = asdict(RadarLead())
    assert rs.leadTwo.present is False
    assert rs.leadTwo.radarTrackId == -1   # cereal schema default
    assert rs.leadTwo.dRel == 0.0


class TestGetLead(OpenpilotTestCase):
  def test_returns_matched_radar_lead(self):
    tracks = {5: make_track(5, d_rel=50.0, y_rel=0.0, v_rel=0.0)}
    vm = vision_msg(x=50.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    lead = get_lead(V_EGO, True, tracks, vm.modelV2.leadsV3[0], V_EGO, vm.modelV2.leadsV3[0].prob,
                    car.CarParams.new_message(), custom.CarParamsSP.new_message())
    assert isinstance(lead, RadarLead)
    assert lead.present and lead.radar and lead.radarTrackId == 5
    assert abs(lead.dRel - 50.0) < 1e-6

  def test_falls_back_to_vision_when_no_track(self):
    vm = vision_msg(x=60.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    lead = get_lead(V_EGO, True, {}, vm.modelV2.leadsV3[0], V_EGO, vm.modelV2.leadsV3[0].prob,
                    car.CarParams.new_message(), custom.CarParamsSP.new_message())
    assert isinstance(lead, RadarLead)
    assert lead.present and not lead.radar and lead.radarTrackId == -1
    assert abs(lead.dRel - 60.0) < 1e-6


class TestHysteresis(OpenpilotTestCase):
  """Id-keyed hysteresis — the default path (LeadTrackingSpatial off), for stable radar track ids."""
  def test_prev_track_gets_sticky_bonus(self):
    # two near-equal tracks; the held one should win once it gets the hysteresis bonus
    tracks = {1: make_track(1, d_rel=50.0, y_rel=0.3, v_rel=0.0),
              2: make_track(2, d_rel=50.0, y_rel=-0.3, v_rel=0.0)}
    vm = vision_msg(x=50.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    lead = vm.modelV2.leadsV3[0]
    no_hyst = match_vision_to_track(V_EGO, lead, tracks, prev_track_id=None)
    held = match_vision_to_track(V_EGO, lead, tracks, prev_track_id=2)
    assert no_hyst is not None and held is not None
    assert held.identifier == 2                      # the held track sticks
    assert match_vision_to_track(V_EGO, lead, tracks, prev_track_id=1).identifier == 1


class TestSpatialAssociation(OpenpilotTestCase):
  """Spatial/kinematic hysteresis — the LeadTrackingSpatial path (built for the Rivian radar's
  ~60/sec track-id renumbering, selectable on any brand). The matcher anchors on the previous
  emitted lead's (dRel, yRel, vRel) instead of its id. Passed as prev_lead_state; takes precedence
  over (a stale) prev_track_id."""
  def test_prefers_track_nearest_prev_position(self):
    # two equal-probability tracks; the one nearest the previous emitted POSITION wins — id-free
    tracks = {1: make_track(1, d_rel=50.0, y_rel=0.3, v_rel=0.0),
              2: make_track(2, d_rel=50.0, y_rel=-0.3, v_rel=0.0)}
    vm = vision_msg(x=50.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    lead = vm.modelV2.leadsV3[0]
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=(50.0, -0.3, 0.0)).identifier == 2
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=(50.0, 0.3, 0.0)).identifier == 1

  def test_velocity_continuity_breaks_ties(self):
    # two tracks at the same spot but different relative speed; vision can't separate them (wide vStd),
    # so the track whose vRel matches the previous emitted lead's wins (velocity continuity)
    tracks = {1: make_track(1, d_rel=60.0, y_rel=0.0, v_rel=0.0),
              2: make_track(2, d_rel=60.0, y_rel=0.0, v_rel=-5.0)}
    vm = vision_msg(x=60.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO, v_std=8.0)
    lead = vm.modelV2.leadsV3[0]
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=(60.0, 0.0, 0.0)).identifier == 1
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=(60.0, 0.0, -5.0)).identifier == 2

  def test_spatial_ignores_stale_id(self):
    # the id-churn fix: the previously-held id has vanished (renumbered). With a spatial anchor, the
    # match is by position regardless of the (now stale) prev_track_id — so the renumbered lead is held.
    tracks = {3: make_track(3, d_rel=50.0, y_rel=0.3, v_rel=0.0),   # same car, renumbered to id 3
              2: make_track(2, d_rel=50.0, y_rel=-0.3, v_rel=0.0)}
    vm = vision_msg(x=50.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    lead = vm.modelV2.leadsV3[0]
    out = match_vision_to_track(V_EGO, lead, tracks, prev_track_id=99, prev_lead_state=(50.0, 0.3, 0.0))
    assert out.identifier == 3                       # picked by position; the stale id 99 is ignored

  def test_spatial_toggle(self):
    # LeadTrackingSpatial param → spatial anchor; default off → id-keyed stickiness.
    # Params writes are async — block until persisted so the immediate read-back sees them.
    rd = make_radard()
    assert rd.use_spatial_assoc is False
    rd.params.put_bool("LeadTrackingSpatial", True, block=True)
    rd._read_lead_tracking_params()
    assert rd.use_spatial_assoc is True
    rd.params.put_bool("LeadTrackingSpatial", False, block=True)
    rd._read_lead_tracking_params()
    assert rd.use_spatial_assoc is False

  def test_update_populates_spatial_anchor(self):
    # end-to-end: with spatial matching on, radard records the emitted lead's (dRel, yRel, vRel)
    # as the next anchor
    rd = make_radard()
    rd.use_spatial_assoc = True
    rd.lead_tracking_mode = 2
    sm = FakeSM(model_msg(VIS_TWO_FAR), carstate_msg())
    rd.update(sm, live_tracks([(100, 95.0, 0.0, 0.0), (101, 120.0, 0.0, 0.0)]))
    assert rd.prev_lead_state[0] is not None and abs(rd.prev_lead_state[0][0] - 95.0) < 1e-6

  def test_length_scales_are_sweepable_instance_attrs(self):
    # the RC length scales follow the coast-threshold pattern: module-const defaults + instance attrs
    rd = make_radard()
    assert rd.spatial_pos_rc == SPATIAL_POS_RC and rd.spatial_vel_rc == SPATIAL_VEL_RC

  def test_velocity_length_scale_honored(self):
    # widening vel_rc weakens velocity-continuity: anchored on the slower track, the default sharp
    # vel_rc holds it, but a wide vel_rc lets the vision-speed-matching track win on prob instead
    tracks = {1: make_track(1, d_rel=60.0, y_rel=0.0, v_rel=0.0),
              2: make_track(2, d_rel=60.0, y_rel=0.0, v_rel=-5.0)}
    vm = vision_msg(x=60.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO, v_std=8.0)
    lead = vm.modelV2.leadsV3[0]
    anchor = (60.0, 0.0, -5.0)   # anchored on the slower track's speed
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=anchor, vel_rc=1.0).identifier == 2
    assert match_vision_to_track(V_EGO, lead, tracks, prev_lead_state=anchor, vel_rc=100.0).identifier == 1


class TestVisionCoast(OpenpilotTestCase):
  def _setup_held(self, rd):
    rd.v_ego = V_EGO
    rd.prev_lead_track_id[0] = 100
    rd.prev_lead_yRel[0] = 0.0
    rd.prev_vision_y[0] = 0.0
    rd.last_good_lead[0] = RadarLead(present=True, radar=True, dRel=95.0, yRel=0.0,
                                     vLead=V_EGO, radarTrackId=100)

  def test_holds_through_phantom(self):
    rd = make_radard()
    self._setup_held(rd)
    # phantom: new id, much closer & slower, while vision still sees the far/fast lead
    phantom = RadarLead(present=True, radar=True, dRel=77.0, yRel=-3.0, vLead=8.0, radarTrackId=200)
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    out, coasting = rd._vision_coast(0, phantom, vm.modelV2.leadsV3[0])
    assert coasting is True
    assert out.radarTrackId == 100 and abs(out.dRel - 95.0) < 1e-6   # held the last-good lead

  def test_passes_real_lead_and_stores_it(self):
    rd = make_radard()
    self._setup_held(rd)
    real = RadarLead(present=True, radar=True, dRel=96.0, yRel=0.0, vLead=V_EGO, radarTrackId=100)
    vm = vision_msg(x=96.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    out, coasting = rd._vision_coast(0, real, vm.modelV2.leadsV3[0])
    assert coasting is False
    assert out is real                                   # emitted unchanged
    assert rd.last_good_lead[0].dRel == 96.0             # and remembered as last-good

  def test_no_coast_when_vision_agrees_with_close_slow(self):
    # a genuinely close/slow lead the camera ALSO sees must not be coasted through
    rd = make_radard()
    self._setup_held(rd)
    real_slow = RadarLead(present=True, radar=True, dRel=77.0, yRel=0.0, vLead=8.0, radarTrackId=200)
    vm = vision_msg(x=77.0 + RADAR_TO_CAMERA, y=0.0, v=8.0)   # vision agrees: close & slow
    out, coasting = rd._vision_coast(0, real_slow, vm.modelV2.leadsV3[0])
    assert coasting is False and out is real_slow

  def test_close_cut_in_failsafe_suppresses_coast(self):
    # a very close radar return (< coast_near_dRel) that LOOKS like a phantom (new id, slow, vision
    # still sees the far/fast lead) must NEVER be coasted — it's a hazard (e.g. a side cut-in)
    rd = make_radard()
    self._setup_held(rd)
    assert rd.coast_near_dRel == 40.0
    cut_in = RadarLead(present=True, radar=True, dRel=18.0, yRel=-3.0, vLead=8.0, radarTrackId=200)
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)   # vision still sees the far/fast lead
    out, coasting = rd._vision_coast(0, cut_in, vm.modelV2.leadsV3[0])
    assert coasting is False                 # failsafe overrode the phantom verdict
    assert out is cut_in and out.dRel == 18.0   # emitted the close lead, did not hold last-good

  def test_phantom_just_beyond_near_threshold_still_coasts(self):
    # the failsafe must not kill the benefit: a phantom just past coast_near_dRel still coasts
    rd = make_radard()
    self._setup_held(rd)
    far_phantom = RadarLead(present=True, radar=True, dRel=55.0, yRel=-3.0, vLead=8.0, radarTrackId=200)
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    out, coasting = rd._vision_coast(0, far_phantom, vm.modelV2.leadsV3[0])
    assert coasting is True and out.radarTrackId == 100   # held last-good, as before (beyond the floor)

  def test_last_good_refreshes_then_clears(self):
    # last-good must never go stale: refresh it to the current non-phantom lead every frame, and
    # clear it to None when there's no radar lead (so we never coast onto a stale value)
    rd = make_radard()
    rd.v_ego = V_EGO
    rd.prev_lead_track_id[0] = 100   # same id below → no corroborator → not a phantom
    # a non-phantom lead that speed-disagrees with vision is now stored fresh (was skipped before)
    lead = RadarLead(present=True, radar=True, dRel=80.0, yRel=0.0, vLead=8.0, radarTrackId=100)
    vm = vision_msg(x=80.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)   # vision 25 vs lead 8 → speed disagrees
    rd._vision_coast(0, lead, vm.modelV2.leadsV3[0])
    assert rd.last_good_lead[0] is not None and rd.last_good_lead[0].dRel == 80.0
    # next frame has no radar lead → last-good cleared, not left stale
    rd._vision_coast(0, RadarLead(), vm.modelV2.leadsV3[0])
    assert rd.last_good_lead[0] is None


class TestIsPhantom(OpenpilotTestCase):
  """The pure per-frame phantom predicate (no cap, no state mutation) — what the harnesses label
  frames with. Each case isolates one gate."""
  def _rd(self):
    rd = make_radard()
    rd.v_ego = V_EGO
    rd.prev_lead_track_id[0] = 100
    rd.prev_lead_yRel[0] = 0.0
    rd.prev_vision_y[0] = 0.0
    return rd

  def _lead(self, dRel=80.0, vLead=8.0, radarTrackId=200, yRel=0.0):
    # a far (>coast_near_dRel), speed-disagreeing radar lead with a changed id, by default a phantom
    return RadarLead(present=True, radar=True, dRel=dRel, yRel=yRel, vLead=vLead, radarTrackId=radarTrackId)

  def test_phantom_true_speed_disagree_plus_id_change(self):
    rd = self._rd()
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)   # vision: far, fast
    assert rd.is_phantom(0, self._lead(), vm.modelV2.leadsV3[0]) is True

  def test_not_phantom_when_speed_agrees(self):
    rd = self._rd()
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    assert rd.is_phantom(0, self._lead(vLead=V_EGO), vm.modelV2.leadsV3[0]) is False

  def test_not_phantom_without_a_corroborator(self):
    # speed disagrees but the id is unchanged (100) and no lateral jump → not a phantom
    rd = self._rd()
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    assert rd.is_phantom(0, self._lead(radarTrackId=100), vm.modelV2.leadsV3[0]) is False

  def test_not_phantom_when_close_failsafe(self):
    rd = self._rd()
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    assert rd.is_phantom(0, self._lead(dRel=18.0), vm.modelV2.leadsV3[0]) is False

  def test_not_phantom_when_vision_unconfident(self):
    rd = self._rd()
    vm = vision_msg(x=95.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO, prob=0.3)
    assert rd.is_phantom(0, self._lead(), vm.modelV2.leadsV3[0]) is False


class TestHandoffSmoothing(OpenpilotTestCase):
  """Tier-3 radar→vision handoff coast (_is_handoff_drop): when the emitted lead drops from a radar
  track to the noisier vision fallback (match rejected the track — overwhelmingly on distance), hold
  the last-good radar lead across the excursion. Safety: obey vision when it reports a genuinely
  NEARER lead (a real cut-in / hard decel) — never delay a real handoff."""
  def _rd(self, held_dRel=120.0):
    rd = make_radard()
    rd.v_ego = V_EGO
    # a recent radar lead in hand to coast with (id 101, far)
    rd.last_good_lead[0] = RadarLead(present=True, radar=True, dRel=held_dRel, vLead=V_EGO, radarTrackId=101)
    return rd

  def _vision_lead(self, dRel):
    # what get_lead emits on a handoff (match → None): present, vision-derived, dRel = vision distance
    return RadarLead(present=True, radar=False, dRel=dRel, vLead=V_EGO, radarTrackId=-1)

  def test_holds_when_vision_corroborates_distance(self):
    # vision still sees the lead at ~the held distance → a radar hiccup → smooth it
    assert self._rd(120.0)._is_handoff_drop(0, self._vision_lead(118.0)) is True

  def test_obeys_genuinely_nearer_vision(self):
    # vision sees the lead far nearer than the held one (60 vs 120) → a real approach → never hold
    assert self._rd(120.0)._is_handoff_drop(0, self._vision_lead(60.0)) is False

  def test_obeys_speed_disagreement(self):
    # the held lead's speed grossly disagrees with the camera's lead — a different vehicle (stale
    # radar track) or a real decel the radar lost → follow vision, never hold (prevents extending a
    # stuck-on-wrong-vehicle gross streak). Distance corroborates (118≈120); only speed disagrees.
    rd = self._rd(120.0)   # held vLead = V_EGO
    slow_vision = RadarLead(present=True, radar=False, dRel=118.0, vLead=V_EGO - 8.0, radarTrackId=-1)
    assert rd._is_handoff_drop(0, slow_vision) is False

  def test_close_cut_in_failsafe(self):
    # a near vision lead (< COAST_NEAR_DREL) is a hazard to respond to, not smooth away
    # (held 42 / vision 38: below the near floor but NOT nearer-gated → isolates the close failsafe)
    assert self._rd(42.0)._is_handoff_drop(0, self._vision_lead(38.0)) is False

  def test_not_handoff_without_last_good(self):
    rd = self._rd()
    rd.last_good_lead[0] = None
    assert rd._is_handoff_drop(0, self._vision_lead(118.0)) is False

  def test_not_handoff_for_radar_lead(self):
    # a radar-derived lead is not a handoff drop — the phantom path owns radar leads
    radar_lead = RadarLead(present=True, radar=True, dRel=118.0, vLead=V_EGO, radarTrackId=200)
    assert self._rd(120.0)._is_handoff_drop(0, radar_lead) is False

  def test_vision_coast_holds_through_handoff(self):
    # integration via _vision_coast: a corroborated handoff drop coasts, holding the last-good radar lead
    rd = self._rd(120.0)
    vm = vision_msg(x=119.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    out, coasting = rd._vision_coast(0, self._vision_lead(119.0), vm.modelV2.leadsV3[0])
    assert coasting is True
    assert out.radar is True and out.radarTrackId == 101 and abs(out.dRel - 120.0) < 1e-6

  def test_vision_coast_obeys_nearer_vision(self):
    # the safety case end-to-end: vision much nearer → no coast, emit the (nearer) vision lead
    rd = self._rd(120.0)
    vm = vision_msg(x=60.0 + RADAR_TO_CAMERA, y=0.0, v=V_EGO)
    out, coasting = rd._vision_coast(0, self._vision_lead(60.0), vm.modelV2.leadsV3[0])
    assert coasting is False
    assert out.radar is False and abs(out.dRel - 60.0) < 1e-6


# --- update() integration: drive the whole RadarD.update path via a minimal fake SubMaster, so
# the per-lead lead_coasting flags (published as radarStateSP for the UI) are exercised. msgq /
# process_replay aborts on macOS, but update() only touches a handful of SubMaster attributes. ---

def model_msg(leads_v3, v_ego=V_EGO):
  msg = messaging.new_message('modelV2')
  msg.modelV2.velocity.x = [v_ego]
  msg.modelV2.leadsV3 = leads_v3
  return msg.modelV2


def lead3(x, y, v, prob=1.0):
  return {'prob': prob, 'x': [x], 'xStd': [2.0], 'y': [y], 'yStd': [1.0], 'v': [v], 'vStd': [1.0], 'a': [0.0]}


def carstate_msg(v_ego=V_EGO):
  msg = messaging.new_message('carState')
  msg.carState.vEgo = v_ego
  return msg.carState


def live_tracks(points):
  """points: list of (trackId, dRel, yRel, vRel) → a radarTracks reader."""
  msg = messaging.new_message('radarTracks')
  msg.radarTracks.points = [{'trackId': t, 'dRel': d, 'yRel': y, 'vRel': v}
                            for (t, d, y, v) in points]
  return msg.radarTracks


class FakeSM:
  def __init__(self, model, carstate):
    self._data = {'modelV2': model, 'carState': carstate}
    self.seen = {'modelV2': True}
    self.logMonoTime = {'modelV2': 1000, 'carState': 1000}
    self.recv_frame = {'carState': 1}

  def __getitem__(self, key):
    return self._data[key]

  def all_checks(self):
    return True


def make_radard_tier(tier):
  rd = make_radard()
  rd.lead_tracking_mode = tier   # update() never reads Params, so a directly-set tier sticks
  return rd


# vision sees two steady, far, same-speed leads the entire time
VIS_TWO_FAR = [lead3(95.0 + RADAR_TO_CAMERA, 0.0, V_EGO), lead3(120.0 + RADAR_TO_CAMERA, 0.0, V_EGO)]


class TestUpdateCoastFlags(OpenpilotTestCase):
  def _prime(self, rd):
    # several frames of two real radar leads matching vision (id 100 near, 101 far) so leadOne
    # locks onto id 100 and it is remembered as last-good
    sm = FakeSM(model_msg(VIS_TWO_FAR), carstate_msg())
    for _ in range(8):
      rd.update(sm, live_tracks([(100, 95.0, 0.0, 0.0), (101, 120.0, 0.0, 0.0)]))

  def test_tier3_update_sets_coast_flag_and_holds_lead(self):
    rd = make_radard_tier(3)
    self._prime(rd)
    assert rd.radar_state.leadOne.radarTrackId == 100
    assert rd.lead_coasting == {0: False, 1: False}
    assert rd.last_good_lead[0] is not None

    # phantom frame: the held tracks (100, 101) drop out and only a close/slow clutter track (200)
    # remains, while vision still sees both far/fast leads. Both leads coast, by the two Tier-3
    # triggers: leadOne matches the clutter 200 → is_phantom → holds 100; leadTwo finds no sane match
    # (200 fails dist_sane vs the 120 m vision lead) → falls to the vision fallback at the SAME 120 m
    # → that's a radar→vision handoff vision corroborates → _is_handoff_drop → holds 101.
    sm = FakeSM(model_msg(VIS_TWO_FAR), carstate_msg())
    rd.update(sm, live_tracks([(200, 77.0, -3.0, -17.0)]))
    assert rd.lead_coasting[0] is True
    assert rd.lead_coasting[1] is True
    assert rd.radar_state.leadOne.radarTrackId == 100             # held last-good, not the phantom
    assert abs(rd.radar_state.leadOne.dRel - 95.0) < 1e-6
    assert rd.radar_state.leadTwo.radarTrackId == 101             # held last-good across the handoff
    assert abs(rd.radar_state.leadTwo.dRel - 120.0) < 1e-6

  def test_tier2_update_never_coasts(self):
    # the same phantom under Tier 2 (no vision-coast) must NOT set the coast flags
    rd = make_radard_tier(2)
    self._prime(rd)
    sm = FakeSM(model_msg(VIS_TWO_FAR), carstate_msg())
    rd.update(sm, live_tracks([(200, 77.0, -3.0, -17.0)]))
    assert rd.lead_coasting == {0: False, 1: False}
    assert rd.radar_state.leadOne.radarTrackId == 200            # the phantom is emitted, not held

  def test_radarstatesp_carries_coast_flags(self):
    # contract: the flags RadarD.publish() copies into radarStateSP survive the cereal roundtrip
    sp = messaging.new_message('radarStateSP')
    sp.radarStateSP.leadOneCoasting = True
    sp.radarStateSP.leadTwoCoasting = False
    out = messaging.log_from_bytes(sp.to_bytes()).radarStateSP
    assert out.leadOneCoasting is True and out.leadTwoCoasting is False


class TestYrelIngestion(OpenpilotTestCase):
  """radard once carried a Rivian-gated 2x lateral correction for the 0.5 scale baked into opendbc's
  rivian/radar_interface.py. The 0.5 was removed upstream (opendbc #3511 / sunnypilot #492), so the
  correction is gone too — guard that ingestion is 1:1 for every brand, so a scale never silently
  reappears on either side."""
  def test_yrel_ingested_unscaled(self):
    for brand in ("rivian", "toyota"):
      cp = car.CarParams.new_message()
      cp.brand = brand
      rd = RadarD(cp, custom.CarParamsSP.new_message(), delay=0.0)
      rd.lead_tracking_mode = 2
      rd.update(FakeSM(model_msg(VIS_TWO_FAR), carstate_msg()), live_tracks([(100, 50.0, 1.0, 0.0)]))
      assert abs(rd.tracks[100].yRel - 1.0) < 1e-6, brand
