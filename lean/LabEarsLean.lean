/-!
# LabEarsLean: L2 Sensory Privacy & Cross-Modal Cavitation Interlock Proofs

Formalizes the `lab-ears` (Layer L2) wet-lab acoustic monitoring guardrails:
1. **Zero Raw-Waveform Persistence**: `rawAudioSaved = false` in shared wet-lab rooms.
2. **Full-Skirt Speech-Band Drop**: Every retained mel/FFT band `[loHz, hiHz]` lies strictly
   outside the `(300, 3400) Hz` human conversation band (`hiHz <= 300 \/ 3400 <= loHz`).
3. **Cross-Modal Cavitation & Stepper-Stall Interlock (Idea #17)**: During an active
   `aspirate` or `dispense` step, high-band (`3.6-7.8 kHz`) cavitation/air-suck spikes or
   low-band (`60-260 Hz`) stepper-stall harmonics deterministically require `haltTriggered = true`
   before air bubbles can be dispensed across an assay plate.
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

/-- Retained acoustic filter frequency support `[loHz, hiHz]` in Hz. -/
structure MelBand where
  loHz : Nat
  hiHz : Nat
  deriving DecidableEq, Repr

/-- Configuration & runtime state of a `lab-ears` acoustic event monitor. -/
structure AcousticEventCheck where
  rawAudioSaved : Bool
  sampleRateHz  : Nat
  retainedBands : List MelBand
  scoreCalibOK  : Bool
  deriving DecidableEq, Repr

/-- True iff the band is well-formed and has zero interior overlap with `(300, 3400) Hz`. -/
def bandOutsideSpeech (b : MelBand) : Bool :=
  decide (b.loHz ≤ b.hiHz ∧ (b.hiHz ≤ 300 ∨ 3400 ≤ b.loHz))

/-- Core L2 `lab-ears` voice-privacy and calibration checker. -/
def checkLabEars (e : AcousticEventCheck) : Option RegimeViolation :=
  firstViolation
    [ (!e.rawAudioSaved,
        { repo := "lab-ears", rule := "zero_raw_audio_persistence",
          detail := "raw microphone waveform retention is forbidden in shared wet-lab rooms" }),
      (decide (8000 ≤ e.sampleRateHz),
        { repo := "lab-ears", rule := "valid_machine_sample_rate",
          detail := "sample rate too low for machine bearing/cavitation harmonic monitoring" }),
      (e.retainedBands.all bandOutsideSpeech,
        { repo := "lab-ears", rule := "full_skirt_speech_band_drop",
          detail := "retained mel filter skirt overlaps human speech band (300-3400 Hz)" }),
      (e.scoreCalibOK,
        { repo := "lab-ears", rule := "calibrated_anomaly_threshold",
          detail := "anomaly detector lacks held-out normal threshold calibration" }) ]

/-- Any frequency strictly inside `(300, 3400) Hz` is excluded by `bandOutsideSpeech`. -/
theorem bandOutsideSpeech_disjoint_interior_speech (b : MelBand) (f : Nat)
    (hb : bandOutsideSpeech b = true) (hlo : 300 < f) (hhi : f < 3400) :
    ¬ (b.loHz ≤ f ∧ f ≤ b.hiHz) := by
  unfold bandOutsideSpeech at hb
  have h := of_decide_eq_true hb
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
  rcases hok with ⟨h1, _, h3, _⟩
  exact ⟨h1, fun b hb => (of_decide_eq_true (h3 b hb)).2⟩

/-- Machine Sample-Rate & Calibration Theorem: Accepted configurations guarantee Nyquist >= 4 kHz
    (`sampleRateHz >= 8000`) and calibrated threshold scoring. -/
theorem checkLabEars_valid_rate_and_calib (e : AcousticEventCheck)
    (hok : checkLabEars e = none) :
    8000 ≤ e.sampleRateHz ∧ e.scoreCalibOK = true := by
  unfold checkLabEars at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.not_eq_true', decide_eq_true_eq] at hok
  rcases hok with ⟨_, h2, _, h4⟩
  exact ⟨h2, h4⟩

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

/-- Combined voice-privacy + cross-modal pipetting fault interlock gate. -/
def checkCrossModalInterlock (c : CrossModalStepCheck) : Option RegimeViolation :=
  match checkLabEars c.event with
  | some v => some v
  | none =>
    firstViolation
      [ (decide (needsInterlockHalt c = true → c.haltTriggered = true),
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
    apply hok.1
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
    apply hok.1
    simp [needsInterlockHalt, hcmd, hstall]

/-! ## Compile-Time `#guard` Witnesses -/

def validAcousticEvent : AcousticEventCheck :=
  { rawAudioSaved := false
    sampleRateHz  := 16000
    retainedBands := [{ loHz := 60, hiHz := 260 }, { loHz := 3600, hiHz := 7800 }]
    scoreCalibOK  := true }

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

-- Speech-band overlap `[250, 500] Hz` or `[1200, 2400] Hz` is rejected
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 250, hiHz := 500 }] }).isSome == true
#guard (checkLabEars { validAcousticEvent with
                       retainedBands := [{ loHz := 1200, hiHz := 2400 }] }).isSome == true

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
