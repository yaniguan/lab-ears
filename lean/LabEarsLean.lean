/-!
# LabEarsLean: L2 Sensory Privacy, Nyquist Realizability & Cross-Modal Interlock Proofs

Formalizes the `lab-ears` (Layer L2) wet-lab acoustic monitoring guardrails:
1. **Zero Raw-Waveform Persistence**: `rawAudioSaved = false` in shared wet-lab rooms.
2. **Non-Empty & Nyquist-Realizable Filter Support**: `retainedBands ≠ []`, `8000 ≤ sampleRateHz`,
   and every retained band `[loHz, hiHz]` satisfies `2 * hiHz ≤ sampleRateHz`.
3. **Finite-Window Hann Skirt Speech-Band Drop**: Every retained mel/FFT band `[loHz, hiHz]`
   broadened by the STFT window half-width `windowSkirtHz` lies strictly outside the
   `(300, 3400) Hz` human conversation band (`hiHz + windowSkirtHz ≤ 300 ∨ 3400 + windowSkirtHz ≤ loHz`).
4. **Conservative Outward Frequency Rounding**: `outwardRoundBand` maps scaled real/rational
   endpoints `[loScaled / scale, hiScaled / scale]` outward to `[⌊lo⌋, ⌈hi⌉]` so integer Hz
   certificates never under-approximate continuous mel filter edges.
5. **Cross-Modal Cavitation & Stepper-Stall Interlock (Idea #17)**: Requires `16000 ≤ sampleRateHz`
   plus both sub-speech (`≤ 300 Hz`) and ultrasonic/high-band (`≥ 3400 Hz`) coverage, and proves
   that high-band cavitation spikes or low-band stepper-stall harmonics during an active
   `aspirate` or `dispense` step deterministically enforce `haltTriggered = true`.
-/
namespace LabEarsLean

/-- Structured boundary violation reporting repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, _⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-- Retained acoustic filter frequency support `[loHz, hiHz]` in integer Hz. -/
structure MelBand where
  loHz : Nat
  hiHz : Nat
  deriving DecidableEq, Repr

/-- Conservative outward rounding of scaled real/rational bounds `[loScaled / scale, hiScaled / scale]`
    into integer Hz `[⌊lo⌋, ⌈hi⌉]` so filter support is never under-approximated. -/
def outwardRoundBand (loScaled hiScaled scale : Nat) : MelBand :=
  { loHz := loScaled / scale
    hiHz := (hiScaled + scale - 1) / scale }

/-- Configuration & runtime state of a `lab-ears` acoustic event monitor. -/
structure AcousticEventCheck where
  rawAudioSaved : Bool
  sampleRateHz  : Nat
  retainedBands : List MelBand
  scoreCalibOK  : Bool
  windowSkirtHz : Nat := 0
  deriving DecidableEq, Repr

/-- True iff the nominal band is well-formed and has zero interior overlap with `(300, 3400) Hz`. -/
def bandOutsideSpeech (b : MelBand) : Bool :=
  decide (b.loHz ≤ b.hiHz ∧ (b.hiHz ≤ 300 ∨ 3400 ≤ b.loHz))

/-- True iff the band's upper edge lies at or below the Nyquist limit `sampleRateHz / 2`. -/
def bandBelowNyquist (sampleRateHz : Nat) (b : MelBand) : Bool :=
  decide (2 * b.hiHz ≤ sampleRateHz)

/-- True iff the band is well-formed and its finite-STFT skirt `[loHz - skirtHz, hiHz + skirtHz]`
    has zero interior overlap with `(300, 3400) Hz`. -/
def bandWithSkirtOutsideSpeech (skirtHz : Nat) (b : MelBand) : Bool :=
  decide (b.loHz ≤ b.hiHz ∧ (b.hiHz + skirtHz ≤ 300 ∨ 3400 + skirtHz ≤ b.loHz))

/-- Core L2 `lab-ears` voice-privacy, Nyquist realizability, and calibration checker. -/
def checkLabEars (e : AcousticEventCheck) : Option RegimeViolation :=
  firstViolation
    [ (!e.rawAudioSaved,
        { repo := "lab-ears", rule := "zero_raw_audio_persistence",
          detail := "raw microphone waveform retention is forbidden in shared wet-lab rooms" }),
      (decide (8000 ≤ e.sampleRateHz),
        { repo := "lab-ears", rule := "valid_machine_sample_rate",
          detail := "sample rate too low for machine bearing/cavitation harmonic monitoring" }),
      (!e.retainedBands.isEmpty,
        { repo := "lab-ears", rule := "nonempty_retained_bands",
          detail := "monitor must retain at least one non-speech machine frequency band" }),
      (e.retainedBands.all (bandBelowNyquist e.sampleRateHz),
        { repo := "lab-ears", rule := "nyquist_band_realizable",
          detail := "retained band upper edge exceeds Nyquist frequency (sampleRateHz / 2)" }),
      (e.retainedBands.all (bandWithSkirtOutsideSpeech e.windowSkirtHz),
        { repo := "lab-ears", rule := "full_skirt_speech_band_drop",
          detail := "retained mel/STFT filter skirt overlaps human speech band (300-3400 Hz)" }),
      (e.scoreCalibOK,
        { repo := "lab-ears", rule := "calibrated_anomaly_threshold",
          detail := "anomaly detector lacks held-out normal threshold calibration" }) ]

/-- Skirt-aware speech exclusion strictly implies nominal `bandOutsideSpeech`. -/
theorem bandWithSkirt_implies_bandOutsideSpeech (skirtHz : Nat) (b : MelBand)
    (hb : bandWithSkirtOutsideSpeech skirtHz b = true) :
    bandOutsideSpeech b = true := by
  unfold bandWithSkirtOutsideSpeech at hb
  unfold bandOutsideSpeech
  have h := of_decide_eq_true hb
  apply decide_eq_true
  omega

/-- Any frequency strictly inside `(300, 3400) Hz` is excluded by `bandOutsideSpeech`. -/
theorem bandOutsideSpeech_disjoint_interior_speech (b : MelBand) (f : Nat)
    (hb : bandOutsideSpeech b = true) (hlo : 300 < f) (hhi : f < 3400) :
    ¬ (b.loHz ≤ f ∧ f ≤ b.hiHz) := by
  unfold bandOutsideSpeech at hb
  have h := of_decide_eq_true hb
  omega

/-- Skirt-Margin Disjointness: No frequency `f` strictly inside `(300, 3400) Hz` can fall
    within the broadened finite-window support `[b.loHz - skirtHz, b.hiHz + skirtHz]`. -/
theorem bandWithSkirt_disjoint_speech_margin (skirtHz : Nat) (b : MelBand) (f : Nat)
    (hb : bandWithSkirtOutsideSpeech skirtHz b = true) (hlo : 300 < f) (hhi : f < 3400) :
    ¬ (b.loHz ≤ f + skirtHz ∧ f ≤ b.hiHz + skirtHz) := by
  unfold bandWithSkirtOutsideSpeech at hb
  have h := of_decide_eq_true hb
  omega

/-- Outward Rounding Containment: Every scaled frequency `fScaled ∈ [loScaled, hiScaled]`
    is enclosed by the outward-rounded integer Hz interval `[loHz * scale, hiHz * scale]`. -/
theorem outwardRoundBand_contains_scaled (loScaled hiScaled fScaled scale : Nat)
    (hscale : 0 < scale) (hlo : loScaled ≤ fScaled) (hhi : fScaled ≤ hiScaled) :
    (outwardRoundBand loScaled hiScaled scale).loHz * scale ≤ fScaled ∧
    fScaled ≤ (outwardRoundBand loScaled hiScaled scale).hiHz * scale := by
  unfold outwardRoundBand
  dsimp
  have hdiv_lo : (loScaled / scale) * scale ≤ loScaled := Nat.div_mul_le_self loScaled scale
  have hdiv_mod := Nat.div_add_mod (hiScaled + scale - 1) scale
  have hmod_lt : (hiScaled + scale - 1) % scale < scale := Nat.mod_lt _ hscale
  have hmul_comm : scale * ((hiScaled + scale - 1) / scale) =
      ((hiScaled + scale - 1) / scale) * scale := Nat.mul_comm _ _
  omega

/-- Outward Rounding Speech-Safety: Certifying `outwardRoundBand loScaled hiScaled scale`
    guarantees exact sub-Hz speech-band separation at arbitrary resolution `scale > 0`. -/
theorem outwardRoundBand_disjoint_scaled_speech (loScaled hiScaled scale skirtHz : Nat)
    (hscale : 0 < scale)
    (hok : bandWithSkirtOutsideSpeech skirtHz (outwardRoundBand loScaled hiScaled scale) = true) :
    hiScaled + skirtHz * scale ≤ 300 * scale ∨
    3400 * scale + skirtHz * scale ≤ loScaled := by
  unfold bandWithSkirtOutsideSpeech outwardRoundBand at hok
  have h := of_decide_eq_true hok
  dsimp at h
  have hdiv_lo : (loScaled / scale) * scale ≤ loScaled := Nat.div_mul_le_self loScaled scale
  have hdiv_mod := Nat.div_add_mod (hiScaled + scale - 1) scale
  have hmod_lt : (hiScaled + scale - 1) % scale < scale := Nat.mod_lt _ hscale
  have hmul_hi : scale * ((hiScaled + scale - 1) / scale) =
      ((hiScaled + scale - 1) / scale) * scale := Nat.mul_comm _ _
  rcases h.2 with hleft | hright
  · left
    have hmul : (((hiScaled + scale - 1) / scale) + skirtHz) * scale ≤ 300 * scale :=
      Nat.mul_le_mul_right scale hleft
    have hdist : (((hiScaled + scale - 1) / scale) + skirtHz) * scale =
        ((hiScaled + scale - 1) / scale) * scale + skirtHz * scale := Nat.add_mul _ _ _
    omega
  · right
    have hmul : (3400 + skirtHz) * scale ≤ (loScaled / scale) * scale :=
      Nat.mul_le_mul_right scale hright
    have hdist : (3400 + skirtHz) * scale = 3400 * scale + skirtHz * scale := Nat.add_mul _ _ _
    omega

/-- Voice-Privacy Theorem: Any accepted `lab-ears` monitor configuration drops raw audio
    AND guarantees zero overlap with `(300, 3400) Hz` across every retained mel filter skirt. -/
theorem checkLabEars_strips_speech (e : AcousticEventCheck)
    (hok : checkLabEars e = none) :
    e.rawAudioSaved = false ∧
    ∀ b ∈ e.retainedBands, b.hiHz ≤ 300 ∨ 3400 ≤ b.loHz := by
  unfold checkLabEars at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.not_eq_true', List.all_eq_true] at hok
  rcases hok with ⟨h1, _, _, _, h5, _⟩
  refine ⟨h1, fun b hb => ?_⟩
  have hsk := of_decide_eq_true (h5 b hb)
  omega

/-- Nyquist & Finite-Window Skirt Theorem: Accepted configurations have non-empty retained
    bands, satisfy `2 * b.hiHz ≤ sampleRateHz` for every band, and maintain a full
    `windowSkirtHz` guard margin outside `(300, 3400) Hz`. -/
theorem checkLabEars_nyquist_and_skirt_bounds (e : AcousticEventCheck)
    (hok : checkLabEars e = none) :
    e.retainedBands ≠ [] ∧
    ∀ b ∈ e.retainedBands,
      2 * b.hiHz ≤ e.sampleRateHz ∧
      (b.hiHz + e.windowSkirtHz ≤ 300 ∨
       (3400 + e.windowSkirtHz ≤ b.loHz ∧ 3400 ≤ b.loHz - e.windowSkirtHz)) := by
  unfold checkLabEars at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.not_eq_true', List.all_eq_true] at hok
  rcases hok with ⟨_, _, hne, hnyq, hsk, _⟩
  refine ⟨?_, fun b hb => ?_⟩
  · intro hemp
    simp [hemp] at hne
  · have hb_nyq : 2 * b.hiHz ≤ e.sampleRateHz := of_decide_eq_true (hnyq b hb)
    have hb_sk := of_decide_eq_true (hsk b hb)
    refine ⟨hb_nyq, ?_⟩
    omega

/-- Machine Sample-Rate & Calibration Theorem: Accepted configurations guarantee Nyquist >= 4 kHz
    (`sampleRateHz >= 8000`) and calibrated threshold scoring. -/
theorem checkLabEars_valid_rate_and_calib (e : AcousticEventCheck)
    (hok : checkLabEars e = none) :
    8000 ≤ e.sampleRateHz ∧ e.scoreCalibOK = true := by
  unfold checkLabEars at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.not_eq_true', decide_eq_true_eq] at hok
  rcases hok with ⟨_, h2, _, _, _, h6⟩
  exact ⟨h2, h6⟩

/-- Cross-modal step-indexed pipetting cavitation and stepper-stall check (Idea #17). -/
structure CrossModalStepCheck where
  event               : AcousticEventCheck
  commandActive       : Bool
  hiBandRatioPermille : Nat   -- 3.6-7.8 kHz cavitation/air-suck ratio vs rheology prior
  hiBandMaxPermille   : Nat   -- allowable cavitation ratio ceiling
  loBandStallPermille : Nat   -- 60-260 Hz stepper-stall harmonic ratio
  loBandMaxPermille   : Nat   -- allowable stepper-stall harmonic ceiling
  haltTriggered       : Bool
  deriving DecidableEq, Repr

/-- Whether an active command window exhibits cavitation air-suck or stepper stall. -/
def needsInterlockHalt (c : CrossModalStepCheck) : Bool :=
  c.commandActive &&
    (decide (c.hiBandMaxPermille < c.hiBandRatioPermille) ||
     decide (c.loBandMaxPermille < c.loBandStallPermille))

/-- Combined voice-privacy + Nyquist + dual-band + cross-modal pipetting fault interlock gate. -/
def checkCrossModalInterlock (c : CrossModalStepCheck) : Option RegimeViolation :=
  match checkLabEars c.event with
  | some v => some v
  | none =>
    firstViolation
      [ (decide (16000 ≤ c.event.sampleRateHz),
          { repo := "lab-ears", rule := "cross_modal_cavitation_sample_rate",
            detail := "cross-modal cavitation monitoring requires sampleRateHz >= 16000 Hz" }),
        (c.event.retainedBands.any (fun b => decide (b.hiHz ≤ 300)),
          { repo := "lab-ears", rule := "low_stall_band_present",
            detail := "cross-modal interlock requires a retained sub-300 Hz stepper-stall band" }),
        (c.event.retainedBands.any (fun b => decide (3400 ≤ b.loHz)),
          { repo := "lab-ears", rule := "high_cavitation_band_present",
            detail := "cross-modal interlock requires a retained >= 3400 Hz cavitation band" }),
        (decide (needsInterlockHalt c = true → c.haltTriggered = true),
          { repo := "lab-ears", rule := "cross_modal_cavitation_or_stall_unhalted",
            detail := "dry-aspiration cavitation or stepper stall detected without run halt" }),
        (decide (c.commandActive = false → c.haltTriggered = false),
          { repo := "lab-ears", rule := "no_spurious_idle_halt",
            detail := "interlock asserted run halt outside an active aspirate/dispense window" }) ]

/-- Cross-Modal Privacy Inheritance: Every certified cross-modal step strips speech bands
    and persists zero raw audio. -/
theorem crossModalInterlock_preserves_voice_privacy (c : CrossModalStepCheck)
    (hok : checkCrossModalInterlock c = none) :
    c.event.rawAudioSaved = false ∧
    ∀ b ∈ c.event.retainedBands, b.hiHz ≤ 300 ∨ 3400 ≤ b.loHz := by
  unfold checkCrossModalInterlock at hok
  cases hev : checkLabEars c.event with
  | some v => simp [hev] at hok
  | none => exact checkLabEars_strips_speech c.event hev

/-- Cross-Modal Nyquist & Dual-Band Coverage: Every certified cross-modal step runs at
    `sampleRateHz >= 16000`, covers both sub-speech and high-frequency cavitation bands,
    and respects Nyquist on every retained band. -/
theorem crossModalInterlock_nyquist_and_dual_bands (c : CrossModalStepCheck)
    (hok : checkCrossModalInterlock c = none) :
    16000 ≤ c.event.sampleRateHz ∧
    (∃ b ∈ c.event.retainedBands, b.hiHz ≤ 300) ∧
    (∃ b ∈ c.event.retainedBands, 3400 ≤ b.loHz) ∧
    (∀ b ∈ c.event.retainedBands, 2 * b.hiHz ≤ c.event.sampleRateHz) := by
  unfold checkCrossModalInterlock at hok
  cases hev : checkLabEars c.event with
  | some v => simp [hev] at hok
  | none =>
    simp only [hev, firstViolation_cons, firstViolation_nil, and_true,
               List.any_eq_true, decide_eq_true_eq] at hok
    rcases hok with ⟨hsr, hlo, hhi, _, _⟩
    have hnyq := (checkLabEars_nyquist_and_skirt_bounds c.event hev).2
    exact ⟨hsr, hlo, hhi, fun b hb => (hnyq b hb).1⟩

/-- Dry-Aspiration Halt Soundness: Any high-band (`3.6-7.8 kHz`) cavitation excess during an
    active pipetting window forces `haltTriggered = true` before plate dispense. -/
theorem crossModalInterlock_halts_dry_aspiration (c : CrossModalStepCheck)
    (hok : checkCrossModalInterlock c = none)
    (hcmd : c.commandActive = true)
    (hcav : c.hiBandMaxPermille < c.hiBandRatioPermille) :
    c.haltTriggered = true := by
  unfold checkCrossModalInterlock at hok
  cases hev : checkLabEars c.event with
  | some v => simp [hev] at hok
  | none =>
    simp only [hev, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    rcases hok with ⟨_, _, _, hhalt, _⟩
    apply hhalt
    simp [needsInterlockHalt, hcmd, hcav]

/-- Stepper-Stall Halt Soundness: Any low-band (`60-260 Hz`) stepper-stall harmonic excess
    during an active pipetting window forces `haltTriggered = true`. -/
theorem crossModalInterlock_halts_stepper_stall (c : CrossModalStepCheck)
    (hok : checkCrossModalInterlock c = none)
    (hcmd : c.commandActive = true)
    (hstall : c.loBandMaxPermille < c.loBandStallPermille) :
    c.haltTriggered = true := by
  unfold checkCrossModalInterlock at hok
  cases hev : checkLabEars c.event with
  | some v => simp [hev] at hok
  | none =>
    simp only [hev, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    rcases hok with ⟨_, _, _, hhalt, _⟩
    apply hhalt
    simp [needsInterlockHalt, hcmd, hstall]

/-! ## Compile-Time `#guard` Witnesses -/

def validAcousticEvent : AcousticEventCheck :=
  { rawAudioSaved := false
    sampleRateHz  := 16000
    retainedBands := [{ loHz := 60, hiHz := 260 }, { loHz := 3600, hiHz := 7800 }]
    scoreCalibOK  := true
    windowSkirtHz := 32 }

def validNormalStep : CrossModalStepCheck :=
  { event               := validAcousticEvent
    commandActive       := true
    hiBandRatioPermille := 1150
    hiBandMaxPermille   := 2200
    loBandStallPermille := 980
    loBandMaxPermille   := 2000
    haltTriggered       := false }

def validDryAspirationHalted : CrossModalStepCheck :=
  { validNormalStep with hiBandRatioPermille := 3850, haltTriggered := true }

-- Valid privacy-preserving acoustic event & normal step pass
#guard checkLabEars validAcousticEvent == none
#guard checkCrossModalInterlock validNormalStep == none
#guard checkCrossModalInterlock validDryAspirationHalted == none

-- Raw audio persistence is rejected
#guard (checkLabEars { validAcousticEvent with rawAudioSaved := true }).isSome == true

-- Empty retainedBands is rejected
#guard (checkLabEars { validAcousticEvent with retainedBands := [] }).isSome == true

-- 8 kHz sample rate is rejected when retaining 3.6-7.8 kHz cavitation band (above 4 kHz Nyquist)
#guard (checkLabEars { validAcousticEvent with sampleRateHz := 8000 }).isSome == true
#guard (checkCrossModalInterlock { validNormalStep with
                                   event := { validAcousticEvent with
                                              sampleRateHz := 8000,
                                              retainedBands := [{ loHz := 60, hiHz := 260 }] } }).isSome == true

-- Speech-band overlap `[250, 500] Hz` or `[1200, 2400] Hz` is rejected
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 250, hiHz := 500 }] }).isSome == true
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 1200, hiHz := 2400 }] }).isSome == true

-- Finite-window Hann skirt bleed (`285 + 32 = 317 > 300` or `3420 < 3400 + 32`) is rejected
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 60, hiHz := 285 }] }).isSome == true
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 3420, hiHz := 7800 }] }).isSome == true

-- Outward rounding catches sub-Hz boundary encroachments (`300.4 Hz` -> 301, `3399.6 Hz` -> 3399)
#guard outwardRoundBand 50000 300400 1000 == { loHz := 50, hiHz := 301 }
#guard outwardRoundBand 3399600 7800000 1000 == { loHz := 3399, hiHz := 7800 }
#guard (checkLabEars { validAcousticEvent with
                       windowSkirtHz := 0,
                       retainedBands := [outwardRoundBand 50000 300400 1000] }).isSome == true
#guard (checkLabEars { validAcousticEvent with
                       windowSkirtHz := 0,
                       retainedBands := [outwardRoundBand 3399600 7800000 1000] }).isSome == true

-- Outward-rounded speech-dropped mel bands `[0.0, 254.544] Hz` & `[3464.967, 8000.0] Hz` pass with 32 Hz skirt
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [outwardRoundBand 0 254544 1000,
                                         outwardRoundBand 3464967 8000000 1000] }) == none

-- Unhalted dry-aspiration cavitation during active command is rejected
#guard (checkCrossModalInterlock { validNormalStep with
                                   hiBandRatioPermille := 3850,
                                   haltTriggered := false }).isSome == true

-- Unhalted stepper stall during active command is rejected
#guard (checkCrossModalInterlock { validNormalStep with
                                   loBandStallPermille := 2750,
                                   haltTriggered := false }).isSome == true

-- Spurious halt during idle transit (`commandActive = false`) is rejected
#guard (checkCrossModalInterlock { validNormalStep with
                                   commandActive := false,
                                   haltTriggered := true }).isSome == true

end LabEarsLean
