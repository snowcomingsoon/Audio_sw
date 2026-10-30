#![forbid(unsafe_code)]

use eframe::egui::{
    self, Align2, Color32, CornerRadius, Event, FontId, Frame, Id, Key, Margin, Order, Pos2, Rect,
    RichText, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use crate::dsp::{DspParams, FilterKind};
use crate::ui_widgets::{brighten, pill_button, StudioPalette};

/// Profilo sorgente di partenza scelto dall'utente al Passo 2.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceProfile {
    SalaProveLive,
    AutoEqIntelligente,
    VoceAcustico,
    DemoMixPotenziato,
}

impl SourceProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::SalaProveLive => "Sala Prove / Live",
            Self::AutoEqIntelligente => "Analisi Auto-EQ",
            Self::VoceAcustico => "Voce / Acustico",
            Self::DemoMixPotenziato => "Demo / Brano Finito",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::SalaProveLive => "Toglie rimbombo e suono chiuso",
            Self::AutoEqIntelligente => "Calcola da solo la curva ideale",
            Self::VoceAcustico => "Calore e nitidezza vocale",
            Self::DemoMixPotenziato => "Spinta, corpo e mastering",
        }
    }

    pub fn accent(self) -> Color32 {
        match self {
            Self::SalaProveLive => StudioPalette::ELECTRIC_CYAN,
            Self::AutoEqIntelligente => StudioPalette::EMERALD,
            Self::VoceAcustico => StudioPalette::STUDIO_VIOLET,
            Self::DemoMixPotenziato => StudioPalette::WARM_AMBER,
        }
    }
}

/// Modalità dedicata di controllo e spinta delle basse frequenze (Bass Boost) al Passo 2.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BassBoostMode {
    AntiRimbombo,
    Bilanciato,
    BassBoostPunch,
    MegaSubBoost,
}

impl BassBoostMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AntiRimbombo => "Asciutti",
            Self::Bilanciato => "Bilanciati",
            Self::BassBoostPunch => "Bass Boost",
            Self::MegaSubBoost => "Mega Sub Boost",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::AntiRimbombo => "Anti-Rimbombo",
            Self::Bilanciato => "Naturali",
            Self::BassBoostPunch => "+Punch & Corpo",
            Self::MegaSubBoost => "+Bassi Profondi",
        }
    }

    pub fn tech_hint(self) -> &'static str {
        match self {
            Self::AntiRimbombo => "HPF 62Hz • -5.5dB Mud",
            Self::Bilanciato => "HPF 40Hz • +1.5dB Corpo",
            Self::BassBoostPunch => "+5.0dB 62Hz • Tight",
            Self::MegaSubBoost => "+7.5dB Sub • Safe Limit",
        }
    }

    pub fn bar_level(self) -> usize {
        match self {
            Self::AntiRimbombo => 1,
            Self::Bilanciato => 2,
            Self::BassBoostPunch => 3,
            Self::MegaSubBoost => 4,
        }
    }

    pub fn accent(self) -> Color32 {
        match self {
            Self::AntiRimbombo => Color32::from_rgb(56, 189, 248),
            Self::Bilanciato => StudioPalette::ELECTRIC_CYAN,
            Self::BassBoostPunch => StudioPalette::WARM_AMBER,
            Self::MegaSubBoost => StudioPalette::EMERALD,
        }
    }
}

/// Nitidezza e presenza della voce (Passo 3).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocalClarityMode {
    Naturale,
    PrimoPiano,
    ExtraNitida,
}

impl VocalClarityMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Naturale => "Naturale",
            Self::PrimoPiano => "Voce in Primo Piano",
            Self::ExtraNitida => "Extra Nitida",
        }
    }
}

/// Apertura sulle alte frequenze ed Exciter armonico (Passo 3).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AirMode {
    MorbidaCalda,
    CristallinaStudio,
    ExtraAria,
}

impl AirMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MorbidaCalda => "Morbida & Calda",
            Self::CristallinaStudio => "Studio Cristallina",
            Self::ExtraAria => "Extra Aria",
        }
    }
}

/// Spinta dinamica del Bus Compressor, ampiezza stereo e loudness finale (Passo 3).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterPushMode {
    Dinamico,
    StreamingPotente,
    MassimaSpinta,
}

impl MasterPushMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dinamico => "Dinamico",
            Self::StreamingPotente => "Potente (-11 LUFS)",
            Self::MassimaSpinta => "Massima Spinta",
        }
    }
}

/// Snapshot immutabile delle scelte effettuate nel Wizard guidato (riutilizzabile anche a fine separazione AI).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuidedWizardChoices {
    pub split_stems_ai: bool,
    pub source_profile: SourceProfile,
    pub bass_mode: BassBoostMode,
    pub vocal_clarity: VocalClarityMode,
    pub air_brilliance: AirMode,
    pub master_push: MasterPushMode,
}

impl Default for GuidedWizardChoices {
    fn default() -> Self {
        Self {
            split_stems_ai: true,
            source_profile: SourceProfile::SalaProveLive,
            bass_mode: BassBoostMode::BassBoostPunch,
            vocal_clarity: VocalClarityMode::PrimoPiano,
            air_brilliance: AirMode::CristallinaStudio,
            master_push: MasterPushMode::StreamingPotente,
        }
    }
}

/// Risultato operativo completo restituito a `main.rs` alla conferma del Wizard.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct WizardOutcome {
    pub params: DspParams,
    pub use_auto_eq_base: bool,
    pub run_ai_split: bool,
    /// Offset volume in dB per i 4 canali Stem `[Voce, Batteria, Basso, Chitarre & Altro]`.
    pub stem_gains_db: [f32; 4],
    /// Valori percentuali `0..=100` sincronizzabili con le 4 manopole Smart `[Pulizia, Voce, Aria, Spinta]`.
    pub macro_knobs_pct: [f32; 4],
    pub summary_msg: String,
}

#[allow(dead_code)]
pub type WizardCompletionAction = WizardOutcome;

#[allow(dead_code)]
impl GuidedWizardChoices {
    /// Calcola i guadagni dedicati in dB per i 4 Stems `[Voce, Batteria, Basso, Chitarre]`.
    pub fn stem_gains_db(&self) -> [f32; 4] {
        let vocal_db = match self.vocal_clarity {
            VocalClarityMode::Naturale => 0.0_f32,
            VocalClarityMode::PrimoPiano => 1.8_f32,
            VocalClarityMode::ExtraNitida => 3.0_f32,
        };

        let (drums_db, bass_db) = match self.bass_mode {
            BassBoostMode::AntiRimbombo => (0.0_f32, -1.0_f32),
            BassBoostMode::Bilanciato => (0.0_f32, 0.0_f32),
            BassBoostMode::BassBoostPunch => (1.5_f32, 3.5_f32),
            BassBoostMode::MegaSubBoost => (2.5_f32, 5.5_f32),
        };

        let guitars_db = match self.source_profile {
            SourceProfile::VoceAcustico => 0.4_f32,
            _ => 0.0_f32,
        };

        [vocal_db, drums_db, bass_db, guitars_db]
    }

    /// Restituisce le 4 percentuali `(0..=100)` per sincronizzare le manopole Smart dell'EQ Drawer:
    /// `[macro_clean, macro_vocal, macro_air, macro_punch]`.
    pub fn macro_knobs_pct(&self) -> [f32; 4] {
        let clean = match self.bass_mode {
            BassBoostMode::AntiRimbombo => 85.0_f32,
            BassBoostMode::Bilanciato => 55.0_f32,
            BassBoostMode::BassBoostPunch => 35.0_f32,
            BassBoostMode::MegaSubBoost => 30.0_f32,
        };
        let vocal = match self.vocal_clarity {
            VocalClarityMode::Naturale => 40.0_f32,
            VocalClarityMode::PrimoPiano => 72.0_f32,
            VocalClarityMode::ExtraNitida => 92.0_f32,
        };
        let air = match self.air_brilliance {
            AirMode::MorbidaCalda => 30.0_f32,
            AirMode::CristallinaStudio => 68.0_f32,
            AirMode::ExtraAria => 90.0_f32,
        };
        let punch = match self.master_push {
            MasterPushMode::Dinamico => 38.0_f32,
            MasterPushMode::StreamingPotente => 68.0_f32,
            MasterPushMode::MassimaSpinta => 92.0_f32,
        };
        [clean, vocal, air, punch]
    }

    /// Costruisce la base `DspParams` prima dell'applicazione delle scelte personalizzate.
    pub fn base_params_for_profile(&self, base_auto_eq: Option<DspParams>, is_for_stems: bool) -> DspParams {
        if is_for_stems {
            return DspParams::preset_stem_polish();
        }

        match self.source_profile {
            SourceProfile::SalaProveLive => DspParams::preset_sala_prove(),
            SourceProfile::AutoEqIntelligente => {
                base_auto_eq.unwrap_or_else(DspParams::preset_sala_prove)
            }
            SourceProfile::VoceAcustico => {
                let mut p = DspParams::preset_voce_primo_piano();
                p.sub_cut_hz = 48.0;
                p.bands[1].gain_db = -4.8;
                p.bands[2].gain_db = -2.2;
                p.bands[3].gain_db = -4.2;
                p.bands[5].gain_db = -3.0;
                p.stereo_width = 1.14;
                p.comp_drive_db = 2.8;
                p
            }
            SourceProfile::DemoMixPotenziato => {
                let mut p = DspParams::preset_rock_punch();
                p.sub_cut_hz = 34.0;
                p.bands[1].gain_db = -2.8;
                p.bands[2].gain_db = -1.6;
                p.bands[3].gain_db = -2.4;
                p.bands[4].gain_db = 3.2;
                p.bands[5].gain_db = -1.5;
                p.bands[6].gain_db = 3.4;
                p.bands[7].gain_db = 5.2;
                p.dynamic_mud_amount = 0.55;
                p.stereo_width = 1.26;
                p.comp_drive_db = 4.2;
                p
            }
        }
    }

    /// Applica Bass Boost, Nitidezza Voce, Aria e Spinta Master a qualsiasi `DspParams`
    /// (sia per ascolto immediato su traccia singola `is_for_stems = false`, sia sul Master Bus
    /// dei 4 Stems AI separati `is_for_stems = true` evitando doppie equalizzazioni aggressive).
    pub fn apply_choices_to_params(&self, mut p: DspParams, is_for_stems: bool) -> DspParams {
        let fallback = DspParams::preset_sala_prove();
        while p.bands.len() < 8 {
            let idx = p.bands.len();
            p.bands.push(fallback.bands[idx].clone());
        }

        p.bypass = false;
        let is_auto_eq = self.source_profile == SourceProfile::AutoEqIntelligente && !is_for_stems;

        // =====================================================================
        // 1. BASS BOOST & CONTROLLO BASSE FREQUENZE (Zero-Clipping & Anti-Mud)
        // =====================================================================
        match self.bass_mode {
            BassBoostMode::AntiRimbombo => {
                p.sub_cut_hz = if is_for_stems { 54.0 } else { 62.0 };

                p.bands[0].name = "108 Hz";
                p.bands[0].role = "Controllo Risonanza";
                p.bands[0].kind = FilterKind::Peaking;
                p.bands[0].freq = 108.0;
                p.bands[0].gain_db = if is_for_stems { -1.8 } else { -3.6 };
                p.bands[0].q = 2.20;
                p.bands[0].enabled = true;

                p.bands[1].name = "160 Hz";
                p.bands[1].role = "Anti-Rimbombo";
                p.bands[1].kind = FilterKind::Peaking;
                p.bands[1].freq = 160.0;
                p.bands[1].gain_db = if is_for_stems { -3.0 } else { -5.5 };
                p.bands[1].q = 1.50;
                p.bands[1].enabled = true;

                if !is_auto_eq {
                    p.bands[2].freq = 265.0;
                    p.bands[2].gain_db = if is_for_stems { -1.4 } else { -3.2 };
                    p.bands[2].q = 1.40;
                    p.bands[2].enabled = true;
                }

                p.dynamic_mud_enabled = true;
                p.dynamic_mud_amount = if is_for_stems { 0.45 } else { 0.85 };
                p.mono_bass_hz = 145.0;
            }
            BassBoostMode::Bilanciato => {
                p.sub_cut_hz = 40.0;

                p.bands[0].name = "65 Hz";
                p.bands[0].role = "Corpo Bassi";
                p.bands[0].kind = FilterKind::Peaking;
                p.bands[0].freq = 65.0;
                p.bands[0].gain_db = if is_for_stems { 1.0 } else { 1.5 };
                p.bands[0].q = 1.25;
                p.bands[0].enabled = true;

                if is_for_stems {
                    p.bands[1].freq = 165.0;
                    p.bands[1].gain_db = -1.0;
                    p.bands[1].q = 1.50;
                    p.bands[1].enabled = true;
                } else if !is_auto_eq {
                    p.bands[1].freq = 164.0;
                    p.bands[1].gain_db = p.bands[1].gain_db.min(-3.2);
                    p.bands[1].q = 1.50;
                    p.bands[1].enabled = true;
                }

                p.dynamic_mud_enabled = true;
                p.dynamic_mud_amount = if is_for_stems { 0.22 } else { 0.55 };
                p.mono_bass_hz = 132.0;
            }
            BassBoostMode::BassBoostPunch => {
                // Protegge dal sub-rumble sotto 32 Hz lasciando passare tutto il punch di cassa e basso
                p.sub_cut_hz = 32.0;

                p.bands[0].name = "62 Hz";
                p.bands[0].role = "Bass Boost Punch";
                p.bands[0].kind = FilterKind::Peaking;
                p.bands[0].freq = 62.0;
                p.bands[0].gain_db = if is_for_stems { 2.8 } else { 5.0 };
                p.bands[0].q = 1.15;
                p.bands[0].enabled = true;

                p.bands[1].name = "95 Hz";
                p.bands[1].role = "Corpo Basso & Cassa";
                p.bands[1].kind = FilterKind::Peaking;
                p.bands[1].freq = 95.0;
                p.bands[1].gain_db = if is_for_stems { 1.8 } else { 3.2 };
                p.bands[1].q = 1.20;
                p.bands[1].enabled = true;

                // Notch anti-mud mirato a 185 Hz (Q 1.8) per mantenere i bassi scolpiti e mai gonfi
                p.bands[2].name = "185 Hz";
                p.bands[2].role = "Taglio Anti-Mud Tight";
                p.bands[2].kind = FilterKind::Peaking;
                p.bands[2].freq = 185.0;
                p.bands[2].gain_db = if is_for_stems { -2.2 } else { -3.0 };
                p.bands[2].q = 1.80;
                p.bands[2].enabled = true;

                p.dynamic_mud_enabled = true;
                p.dynamic_mud_amount = if is_for_stems { 0.18 } else { 0.35 };
                p.mono_bass_hz = 128.0;
            }
            BassBoostMode::MegaSubBoost => {
                // Estensione sub profonda fino a 28 Hz con focus mono-centrato e protezione soft-knee
                p.sub_cut_hz = 28.0;

                p.bands[0].name = "55 Hz";
                p.bands[0].role = "Mega Sub Boost";
                p.bands[0].kind = FilterKind::Peaking;
                p.bands[0].freq = 55.0;
                p.bands[0].gain_db = if is_for_stems { 4.2 } else { 7.5 };
                p.bands[0].q = 1.00;
                p.bands[0].enabled = true;

                p.bands[1].name = "90 Hz";
                p.bands[1].role = "Mega Bass Punch";
                p.bands[1].kind = FilterKind::Peaking;
                p.bands[1].freq = 90.0;
                p.bands[1].gain_db = if is_for_stems { 2.5 } else { 4.5 };
                p.bands[1].q = 1.15;
                p.bands[1].enabled = true;

                p.bands[2].name = "195 Hz";
                p.bands[2].role = "Anti-Mud Sub Clean";
                p.bands[2].kind = FilterKind::Peaking;
                p.bands[2].freq = 195.0;
                p.bands[2].gain_db = if is_for_stems { -2.8 } else { -3.5 };
                p.bands[2].q = 1.85;
                p.bands[2].enabled = true;

                p.dynamic_mud_enabled = true;
                p.dynamic_mud_amount = if is_for_stems { 0.16 } else { 0.30 };
                p.mono_bass_hz = 135.0;
            }
        }

        // =====================================================================
        // 2. PRESENZA VOCE (1.6 kHz Body + 4.5 kHz Articolazione + De-Harsh)
        // =====================================================================
        match self.vocal_clarity {
            VocalClarityMode::Naturale => {
                if is_for_stems {
                    p.bands[4].freq = 1600.0;
                    p.bands[4].gain_db = 0.0;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 0.0;
                } else if is_auto_eq {
                    p.bands[4].gain_db = (p.bands[4].gain_db - 0.8).clamp(-4.5, 6.0);
                    p.bands[6].gain_db = (p.bands[6].gain_db - 0.6).clamp(-4.0, 5.5);
                } else {
                    p.bands[4].freq = 1580.0;
                    p.bands[4].gain_db = 2.4;
                    p.bands[4].q = 0.90;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 2.2;
                    p.bands[6].q = 1.05;
                }
                p.bands[4].enabled = true;
                p.bands[6].enabled = true;
            }
            VocalClarityMode::PrimoPiano => {
                if is_for_stems {
                    p.bands[4].freq = 1600.0;
                    p.bands[4].gain_db = 1.2;
                    p.bands[4].q = 0.90;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 1.0;
                    p.bands[6].q = 1.05;
                } else if is_auto_eq {
                    p.bands[4].gain_db = (p.bands[4].gain_db + 1.6).clamp(1.2, 8.2);
                    p.bands[6].gain_db = (p.bands[6].gain_db + 1.4).clamp(1.0, 7.2);
                } else {
                    p.bands[4].freq = 1600.0;
                    p.bands[4].gain_db = 5.2;
                    p.bands[4].q = 0.88;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 4.6;
                    p.bands[6].q = 1.00;
                }
                p.bands[4].enabled = true;
                p.bands[6].enabled = true;
            }
            VocalClarityMode::ExtraNitida => {
                if is_for_stems {
                    p.bands[4].freq = 1600.0;
                    p.bands[4].gain_db = 2.0;
                    p.bands[4].q = 0.88;
                    p.bands[5].freq = 2450.0;
                    p.bands[5].gain_db = -1.4;
                    p.bands[5].q = 3.20;
                    p.bands[5].enabled = true;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 1.8;
                    p.bands[6].q = 1.00;
                } else if is_auto_eq {
                    p.bands[4].gain_db = (p.bands[4].gain_db + 3.0).clamp(2.8, 9.2);
                    p.bands[5].gain_db = p.bands[5].gain_db.min(-2.2);
                    p.bands[5].enabled = true;
                    p.bands[6].gain_db = (p.bands[6].gain_db + 2.6).clamp(2.4, 8.2);
                } else {
                    p.bands[4].freq = 1600.0;
                    p.bands[4].gain_db = 6.8;
                    p.bands[4].q = 0.85;
                    p.bands[5].freq = 2400.0;
                    p.bands[5].gain_db = -2.8;
                    p.bands[5].q = 3.30;
                    p.bands[5].enabled = true;
                    p.bands[6].freq = 4500.0;
                    p.bands[6].gain_db = 5.8;
                    p.bands[6].q = 0.95;
                }
                p.bands[4].enabled = true;
                p.bands[6].enabled = true;
            }
        }

        // =====================================================================
        // 3. BRILLANTEZZA ALTI & ARIA STUDIO (8.5 kHz High-Shelf + Exciter)
        // =====================================================================
        p.bands[7].kind = FilterKind::HighShelf;
        p.bands[7].freq = 8500.0;
        p.bands[7].q = 0.707;
        p.bands[7].enabled = true;

        match self.air_brilliance {
            AirMode::MorbidaCalda => {
                p.bands[7].gain_db = if is_for_stems {
                    0.5
                } else if is_auto_eq {
                    (p.bands[7].gain_db - 2.0).clamp(-3.0, 3.8)
                } else {
                    2.4
                };
                p.exciter_drive = if is_for_stems { 0.03 } else { 0.08 };
                p.exciter_enabled = p.exciter_drive > 0.01;
            }
            AirMode::CristallinaStudio => {
                p.bands[7].gain_db = if is_for_stems {
                    2.2
                } else if is_auto_eq {
                    (p.bands[7].gain_db + 1.0).clamp(2.2, 8.5)
                } else {
                    6.8
                };
                p.exciter_drive = if is_for_stems { 0.12 } else { 0.22 };
                p.exciter_enabled = true;
            }
            AirMode::ExtraAria => {
                p.bands[7].gain_db = if is_for_stems {
                    3.8
                } else if is_auto_eq {
                    (p.bands[7].gain_db + 2.8).clamp(4.5, 10.2)
                } else {
                    9.0
                };
                p.exciter_drive = if is_for_stems { 0.20 } else { 0.34 };
                p.exciter_enabled = true;
            }
        }

        // =====================================================================
        // 4. VOLUME MASTER, GLUE COMPRESSOR & PROTEZIONE ANTI-CLIPPING
        // =====================================================================
        p.mid_side_enabled = true;
        p.comp_enabled = true;

        let (base_drive_db, width) = match self.master_push {
            MasterPushMode::Dinamico => (
                if is_for_stems { 0.8 } else { 1.8 },
                if is_for_stems { 1.06 } else { 1.12 },
            ),
            MasterPushMode::StreamingPotente => (
                if is_for_stems { 2.4 } else { 4.0 },
                if is_for_stems { 1.14 } else { 1.24 },
            ),
            MasterPushMode::MassimaSpinta => (
                if is_for_stems { 3.8 } else { 5.8 },
                if is_for_stems { 1.22 } else { 1.34 },
            ),
        };

        // Compensazione intelligente di headroom sul Bus Compressor per preservare punch e 0% distorsione
        let bass_headroom_trim: f64 = match self.bass_mode {
            BassBoostMode::MegaSubBoost => -0.8_f64,
            BassBoostMode::BassBoostPunch => -0.3_f64,
            _ => 0.0_f64,
        };

        p.comp_drive_db = (base_drive_db + bass_headroom_trim).clamp(0.4_f64, 8.5_f64);
        p.stereo_width = width;
        p.ceiling_db = if self.bass_mode == BassBoostMode::MegaSubBoost {
            -0.5
        } else {
            -0.4
        };

        p
    }

    /// Genera direttamente i parametri Master Bus calibrati per i 4 Stems AI separati.
    pub fn build_stem_master_params(&self) -> DspParams {
        self.apply_choices_to_params(DspParams::preset_stem_polish(), true)
    }

    /// Frase di riepilogo chiara per la barra di stato e per il Passo 3 del Wizard.
    pub fn summary_line(&self) -> String {
        let mode_str = if self.split_stems_ai {
            "Separazione AI in 4 tracce"
        } else {
            "Applicazione istantanea su traccia singola"
        };
        let bass_str = match self.bass_mode {
            BassBoostMode::AntiRimbombo => "Bassi Asciutti Anti-Rimbombo",
            BassBoostMode::Bilanciato => "Bassi Bilanciati",
            BassBoostMode::BassBoostPunch => "Bass Boost (+Punch & Corpo)",
            BassBoostMode::MegaSubBoost => "Mega Sub Boost (+Bassi Profondi)",
        };
        format!(
            "{}  •  {}  •  {}  •  {}",
            mode_str,
            self.source_profile.label(),
            bass_str,
            self.vocal_clarity.label()
        )
    }

    /// Costruisce il `WizardOutcome` completo pronto per essere applicato in `main.rs`.
    pub fn build_outcome(&self, base_auto_eq: Option<DspParams>) -> WizardOutcome {
        let use_auto_eq_base = self.source_profile == SourceProfile::AutoEqIntelligente;
        let base = self.base_params_for_profile(base_auto_eq, false);
        let params = self.apply_choices_to_params(base, false);

        WizardOutcome {
            params,
            use_auto_eq_base,
            run_ai_split: self.split_stems_ai,
            stem_gains_db: self.stem_gains_db(),
            macro_knobs_pct: self.macro_knobs_pct(),
            summary_msg: format!("Configurazione applicata: {}", self.summary_line()),
        }
    }
}

/// Stato completo della finestra modale guidata a 3 passi mostrata all'apertura di una traccia.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct GuidedWizardState {
    pub open: bool,
    pub step: u8,
    pub track_name: String,
    pub split_stems_ai: bool,
    pub source_profile: SourceProfile,
    pub bass_mode: BassBoostMode,
    pub vocal_clarity: VocalClarityMode,
    pub air_brilliance: AirMode,
    pub master_push: MasterPushMode,
    /// Ultima configurazione confermata dall'utente (usata per ri-applicare Bass Boost e guadagni
    /// quando la separazione AI in background termina i 4 Stems!).
    pub last_applied: Option<GuidedWizardChoices>,
}

impl Default for GuidedWizardState {
    fn default() -> Self {
        let d = GuidedWizardChoices::default();
        Self {
            open: false,
            step: 1,
            track_name: String::new(),
            split_stems_ai: d.split_stems_ai,
            source_profile: d.source_profile,
            bass_mode: d.bass_mode,
            vocal_clarity: d.vocal_clarity,
            air_brilliance: d.air_brilliance,
            master_push: d.master_push,
            last_applied: None,
        }
    }
}

#[allow(dead_code)]
impl GuidedWizardState {
    /// Apre il Wizard al Passo 1 per una nuova traccia appena caricata, impostando default ottimali.
    pub fn open_for_track(&mut self, track_name: &str) {
        let d = GuidedWizardChoices::default();
        self.open = true;
        self.step = 1;
        self.track_name = track_name.to_string();
        self.split_stems_ai = d.split_stems_ai;
        self.source_profile = d.source_profile;
        self.bass_mode = d.bass_mode;
        self.vocal_clarity = d.vocal_clarity;
        self.air_brilliance = d.air_brilliance;
        self.master_push = d.master_push;
    }

    pub fn current_choices(&self) -> GuidedWizardChoices {
        GuidedWizardChoices {
            split_stems_ai: self.split_stems_ai,
            source_profile: self.source_profile,
            bass_mode: self.bass_mode,
            vocal_clarity: self.vocal_clarity,
            air_brilliance: self.air_brilliance,
            master_push: self.master_push,
        }
    }

    pub fn stem_gains_db(&self) -> [f32; 4] {
        self.current_choices().stem_gains_db()
    }

    pub fn macro_knobs_pct(&self) -> [f32; 4] {
        self.current_choices().macro_knobs_pct()
    }

    pub fn apply_choices_to_params(&self, p: DspParams, is_for_stems: bool) -> DspParams {
        self.current_choices()
            .apply_choices_to_params(p, is_for_stems)
    }

    pub fn build_outcome(&self, base_auto_eq: Option<DspParams>) -> WizardOutcome {
        self.current_choices().build_outcome(base_auto_eq)
    }
}

#[inline]
fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

#[inline]
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0_f32, 1.0_f32);
    let inv = 1.0_f32 - t;
    let r = (a.r() as f32 * inv + b.r() as f32 * t).round() as u8;
    let g = (a.g() as f32 * inv + b.g() as f32 * t).round() as u8;
    let bl = (a.b() as f32 * inv + b.b() as f32 * t).round() as u8;
    let al = (a.a() as f32 * inv + b.a() as f32 * t).round() as u8;
    Color32::from_rgba_unmultiplied(r, g, bl, al)
}

fn consume_key_once(i: &mut egui::InputState, target: Key) -> bool {
    let mut hit = false;
    i.events.retain(|ev| match ev {
        Event::Key {
            key,
            physical_key,
            pressed: true,
            repeat,
            modifiers,
        } if (*key == target || *physical_key == Some(target))
            && !modifiers.command
            && !modifiers.ctrl
            && !modifiers.alt =>
        {
            if !*repeat {
                hit = true;
            }
            false
        }
        _ => true,
    });
    hit
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars || max_chars < 10 {
        return text.to_string();
    }
    let keep_left = (max_chars - 3) / 2;
    let keep_right = max_chars - 3 - keep_left;
    let left: String = text.chars().take(keep_left).collect();
    let right: String = text.chars().skip(count - keep_right).collect();
    format!("{left}...{right}")
}

/// Disegna la finestra modale guidata a 3 passi all'aggiunta di una traccia.
/// Restituisce `Some(GuidedWizardChoices)` nell'istante in cui l'utente conferma al Passo 3.
pub fn draw_guided_wizard_modal(
    ctx: &egui::Context,
    state: &mut GuidedWizardState,
) -> Option<GuidedWizardChoices> {
    if !state.open {
        return None;
    }

    state.step = state.step.clamp(1, 3);

    if ctx.input_mut(|i| consume_key_once(i, Key::Escape)) {
        state.open = false;
        return None;
    }

    let enter_pressed = ctx.input_mut(|i| consume_key_once(i, Key::Enter));

    let screen = ctx.screen_rect();
    let mut skip_requested = false;
    let mut confirm_requested = false;

    // 1. Backdrop scuro satinato (blocca i clic verso i controlli sottostanti)
    egui::Area::new(Id::new("studio_eq_guided_wizard_backdrop"))
        .order(Order::Middle)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(
                rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(5, 8, 15, 206),
            );
        });

    let modal_frame = Frame::window(&ctx.style())
        .fill(Color32::from_rgb(15, 20, 32))
        .corner_radius(CornerRadius::same(16))
        .stroke(Stroke::new(1.2_f32, Color32::from_rgb(46, 62, 92)))
        .inner_margin(Margin::symmetric(24, 20));

    egui::Window::new("Configurazione Guidata Traccia")
        .id(Id::new("studio_eq_guided_wizard_modal"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .order(Order::Foreground)
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .default_width(650.0_f32)
        .min_width(650.0_f32)
        .max_width(650.0_f32)
        .frame(modal_frame)
        .show(ctx, |ui| {
            let content_w = 602.0_f32;
            ui.set_width(content_w);

            // =================================================================
            // HEADER SUPERIORE: Titolo + Badge Nome Traccia + Pulsante Salta
            // =================================================================
            ui.horizontal(|ui| {
                let (dot_rect, _) =
                    ui.allocate_exact_size(Vec2::splat(10.0_f32), Sense::hover());
                let header_accent = match state.step {
                    1 => StudioPalette::STUDIO_VIOLET,
                    2 => StudioPalette::WARM_AMBER,
                    _ => StudioPalette::EMERALD,
                };
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.5_f32, header_accent);

                ui.label(
                    RichText::new("SETUP GUIDATO DEL SUONO")
                        .size(15.5_f32)
                        .strong()
                        .color(StudioPalette::CRISP_TEXT),
                );

                if !state.track_name.is_empty() {
                    let short_track = truncate_middle(&state.track_name, 30);
                    Frame::NONE
                        .fill(StudioPalette::DEEP_OBSIDIAN)
                        .corner_radius(CornerRadius::same(8))
                        .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
                        .inner_margin(Margin::symmetric(10, 4))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(short_track)
                                    .monospace()
                                    .size(11.5_f32)
                                    .color(StudioPalette::ELECTRIC_CYAN),
                            );
                        });
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if pill_button(
                        ui,
                        "✕ Salta",
                        false,
                        StudioPalette::MUTED_LABEL,
                        Vec2::new(78.0_f32, 30.0_f32),
                    )
                    .on_hover_text("Chiudi il menu guidato e mantieni le impostazioni attuali (Esc)")
                    .clicked()
                    {
                        skip_requested = true;
                    }
                });
            });

            ui.add_space(10.0_f32);

            // =================================================================
            // INDICATORE DI AVANZAMENTO: PASSO 1 DI 3 • PASSO 2 DI 3 • PASSO 3 DI 3
            // =================================================================
            draw_step_indicator_strip(ui, state, content_w);

            ui.add_space(14.0_f32);

            // =================================================================
            // CONTENUTO DEL PASSO ATTIVO (1, 2 o 3)
            // =================================================================
            match state.step {
                1 => draw_step_1_split_choice(ui, state, content_w),
                2 => draw_step_2_source_and_bass(ui, state, content_w),
                _ => draw_step_3_vocal_air_master(ui, state, content_w),
            }

            ui.add_space(16.0_f32);

            // Separatore sottile sopra la barra di navigazione
            let (sep_rect, _) =
                ui.allocate_exact_size(Vec2::new(content_w, 1.0_f32), Sense::hover());
            ui.painter()
                .rect_filled(sep_rect, CornerRadius::ZERO, StudioPalette::SUBTLE_HAIRLINE);

            ui.add_space(12.0_f32);

            // =================================================================
            // FOOTER DI NAVIGAZIONE
            // =================================================================
            ui.horizontal(|ui| {
                if state.step > 1 {
                    if pill_button(
                        ui,
                        "<- Indietro",
                        false,
                        StudioPalette::ELECTRIC_CYAN,
                        Vec2::new(112.0_f32, 38.0_f32),
                    )
                    .clicked()
                    {
                        state.step = state.step.saturating_sub(1).max(1);
                    }
                } else {
                    ui.label(
                        RichText::new("Suggerimento: potrai ritoccare l'EQ anche dopo")
                            .size(11.8_f32)
                            .color(StudioPalette::MUTED_LABEL),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if state.step < 3 {
                        let next_accent = if state.step == 1 && state.split_stems_ai {
                            StudioPalette::STUDIO_VIOLET
                        } else {
                            StudioPalette::ELECTRIC_CYAN
                        };
                        if pill_button(
                            ui,
                            "Avanti  ->",
                            true,
                            next_accent,
                            Vec2::new(138.0_f32, 38.0_f32),
                        )
                        .clicked()
                            || enter_pressed
                        {
                            state.step = (state.step + 1).min(3);
                        }
                    } else {
                        let cta_label = if state.split_stems_ai {
                            "Avvia Separazione & Applica"
                        } else {
                            "Applica Subito alla Traccia"
                        };
                        let cta_w = if state.split_stems_ai {
                            238.0_f32
                        } else {
                            220.0_f32
                        };
                        if pill_button(
                            ui,
                            cta_label,
                            true,
                            StudioPalette::EMERALD,
                            Vec2::new(cta_w, 40.0_f32),
                        )
                        .clicked()
                            || enter_pressed
                        {
                            confirm_requested = true;
                        }
                    }
                });
            });
        });

    if skip_requested {
        state.open = false;
        return None;
    }

    if confirm_requested {
        let choices = state.current_choices();
        state.last_applied = Some(choices);
        state.open = false;
        return Some(choices);
    }

    None
}

fn draw_step_indicator_strip(ui: &mut Ui, state: &mut GuidedWizardState, total_w: f32) {
    let steps: [(u8, &str, Color32); 3] = [
        (1, "PASSO 1 DI 3  •  Tracce", StudioPalette::STUDIO_VIOLET),
        (2, "PASSO 2 DI 3  •  Bass Boost", StudioPalette::WARM_AMBER),
        (3, "PASSO 3 DI 3  •  Finitura", StudioPalette::EMERALD),
    ];

    let gap = 8.0_f32;
    let item_w = ((total_w - gap * 2.0_f32) / 3.0_f32).floor();

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for (num, label, accent) in steps {
            let is_current = state.step == num;
            let is_done = state.step > num;
            let (rect, resp) =
                ui.allocate_exact_size(Vec2::new(item_w, 30.0_f32), Sense::click());
            if resp.clicked() {
                state.step = num;
            }
            if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }

            if ui.is_rect_visible(rect) {
                let hover_t = ui
                    .ctx()
                    .animate_bool_responsive(resp.id.with("step_hov"), resp.hovered());
                let painter = ui.painter();
                let radius = CornerRadius::same(8);

                let bg = if is_current {
                    with_alpha(accent, 42)
                } else if is_done {
                    with_alpha(StudioPalette::EMERALD, 22)
                } else {
                    lerp_color(
                        StudioPalette::DEEP_OBSIDIAN,
                        StudioPalette::CARD_GLASS,
                        hover_t,
                    )
                };
                let border = if is_current {
                    brighten(accent, 22)
                } else if is_done {
                    with_alpha(StudioPalette::EMERALD, 130)
                } else {
                    lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, hover_t * 0.45_f32)
                };

                painter.rect_filled(rect, radius, bg);
                painter.rect_stroke(
                    rect,
                    radius,
                    Stroke::new(if is_current { 1.3_f32 } else { 1.0_f32 }, border),
                    StrokeKind::Inside,
                );

                let display = if is_done {
                    format!("✓  {label}")
                } else {
                    label.to_string()
                };
                let text_col = if is_current {
                    StudioPalette::CRISP_TEXT
                } else if is_done {
                    brighten(StudioPalette::EMERALD, 35)
                } else {
                    StudioPalette::MUTED_LABEL
                };
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    display,
                    FontId::proportional(11.6_f32),
                    text_col,
                );
            }
        }
    });
}

fn draw_step_1_split_choice(ui: &mut Ui, state: &mut GuidedWizardState, total_w: f32) {
    ui.label(
        RichText::new("1. Vuoi dividere gli strumenti della traccia?")
            .size(18.0_f32)
            .strong()
            .color(StudioPalette::CRISP_TEXT),
    );
    ui.add_space(2.0_f32);
    ui.label(
        RichText::new(
            "Scegli se separare gli strumenti con l'intelligenza artificiale o intervenire subito sull'intero brano.",
        )
        .size(12.6_f32)
        .color(StudioPalette::MUTED_LABEL),
    );
    ui.add_space(12.0_f32);

    let gap = 14.0_f32;
    let card_w = ((total_w - gap) * 0.5_f32).floor();
    let card_h = 228.0_f32;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;

        if draw_step1_big_card(
            ui,
            Vec2::new(card_w, card_h),
            state.split_stems_ai,
            StudioPalette::STUDIO_VIOLET,
            "CONSIGLIATO • 4 STEMS AI",
            "DIVIDI LE TRACCE CON AI",
            "Consigliato per Band, Sala Prove e Canzoni complete",
            &[
                "• Separa Voce, Batteria, Basso e Chitarre in 4 piste",
                "• Regoli volume, Mute e Solo di ogni strumento",
                "• Usa la tua GPU NVIDIA in automatico",
            ],
        ) {
            state.split_stems_ai = true;
        }

        if draw_step1_big_card(
            ui,
            Vec2::new(card_w, card_h),
            !state.split_stems_ai,
            StudioPalette::ELECTRIC_CYAN,
            "ISTANTANEO • REAL-TIME",
            "TRACCIA UNICA IMMEDIATA",
            "Ideale per ascolto istantaneo, singola voce o strumento",
            &[
                "• Nessuna attesa: risultato in tempo reale",
                "• Equalizzazione, Bass Boost e Mastering immediati",
                "• Puoi sempre separare le tracce più tardi",
            ],
        ) {
            state.split_stems_ai = false;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn draw_step1_big_card(
    ui: &mut Ui,
    size: Vec2,
    selected: bool,
    accent: Color32,
    badge_text: &str,
    title: &str,
    subtitle: &str,
    bullets: &[&str; 3],
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("s1_card_hov"), resp.hovered());
        let sel_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("s1_card_sel"), selected);
        let painter = ui.painter();
        let radius = CornerRadius::same(14);

        // Alone esterno morbido quando selezionato
        if sel_t > 0.01_f32 || hover_t > 0.01_f32 {
            let glow_a = (30.0_f32 * sel_t + 14.0_f32 * hover_t)
                .clamp(0.0_f32, 60.0_f32)
                .round() as u8;
            painter.rect_filled(rect.expand(2.5_f32), CornerRadius::same(16), with_alpha(accent, glow_a));
        }

        let base_bg = lerp_color(
            StudioPalette::DEEP_OBSIDIAN,
            StudioPalette::CARD_GLASS,
            hover_t * 0.65_f32,
        );
        let bg = lerp_color(base_bg, with_alpha(accent, 34), sel_t * 0.65_f32);
        painter.rect_filled(rect, radius, bg);

        let border_col = if selected {
            brighten(accent, 26)
        } else {
            lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, hover_t * 0.55_f32)
        };
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(if selected { 1.8_f32 } else { 1.0_f32 }, border_col),
            StrokeKind::Inside,
        );

        let pad_x = rect.left() + 16.0_f32;
        let mut cur_y = rect.top() + 15.0_f32;

        // Riga superiore: Badge categoria + Checkmark stato
        let badge_font = FontId::proportional(10.5_f32);
        let badge_galley =
            painter.layout_no_wrap(badge_text.to_string(), badge_font.clone(), accent);
        let badge_rect = Rect::from_min_size(
            Pos2::new(pad_x, cur_y),
            Vec2::new(badge_galley.size().x + 14.0_f32, 21.0_f32),
        );
        painter.rect_filled(badge_rect, CornerRadius::same(6), with_alpha(accent, 36));
        painter.rect_stroke(
            badge_rect,
            CornerRadius::same(6),
            Stroke::new(1.0_f32, with_alpha(accent, 120)),
            StrokeKind::Inside,
        );
        painter.text(
            badge_rect.center(),
            Align2::CENTER_CENTER,
            badge_text,
            badge_font,
            brighten(accent, 32),
        );

        // Pillola Checkmark a destra
        let check_center = Pos2::new(rect.right() - 25.0_f32, badge_rect.center().y);
        if selected {
            let chk_rect = Rect::from_center_size(check_center, Vec2::new(26.0_f32, 22.0_f32));
            painter.rect_filled(chk_rect, CornerRadius::same(11), accent);
            painter.text(
                chk_rect.center(),
                Align2::CENTER_CENTER,
                "✓",
                FontId::proportional(13.5_f32),
                StudioPalette::CRISP_TEXT,
            );
        } else {
            painter.circle_stroke(
                check_center,
                9.5_f32,
                Stroke::new(1.2_f32, StudioPalette::SUBTLE_HAIRLINE),
            );
        }

        cur_y += 32.0_f32;

        // Titolo Card
        painter.text(
            Pos2::new(pad_x, cur_y),
            Align2::LEFT_TOP,
            title,
            FontId::proportional(15.5_f32),
            StudioPalette::CRISP_TEXT,
        );
        cur_y += 23.0_f32;

        // Sottotitolo avvolgente su 1-2 righe
        let sub_galley = painter.layout(
            subtitle.to_string(),
            FontId::proportional(12.2_f32),
            if selected {
                Color32::from_rgb(206, 218, 236)
            } else {
                StudioPalette::MUTED_LABEL
            },
            size.x - 32.0_f32,
        );
        let sub_h = sub_galley.size().y;
        painter.galley(Pos2::new(pad_x, cur_y), sub_galley, StudioPalette::MUTED_LABEL);
        cur_y += sub_h + 14.0_f32;

        // Linea divisoria interna
        painter.line_segment(
            [
                Pos2::new(pad_x, cur_y),
                Pos2::new(rect.right() - 16.0_f32, cur_y),
            ],
            Stroke::new(1.0_f32, with_alpha(StudioPalette::SUBTLE_HAIRLINE, 160)),
        );
        cur_y += 11.0_f32;

        // Elenco puntato chiaro
        for bullet in bullets {
            let b_galley = painter.layout(
                (*bullet).to_string(),
                FontId::proportional(12.0_f32),
                Color32::from_rgb(224, 232, 244),
                size.x - 30.0_f32,
            );
            let bh = b_galley.size().y;
            painter.galley(Pos2::new(pad_x, cur_y), b_galley, StudioPalette::CRISP_TEXT);
            cur_y += bh + 6.0_f32;
        }
    }

    resp.clicked()
}

fn draw_step_2_source_and_bass(ui: &mut Ui, state: &mut GuidedWizardState, total_w: f32) {
    ui.label(
        RichText::new("2. Tipo di Audio & Boost dei Bassi")
            .size(18.0_f32)
            .strong()
            .color(StudioPalette::CRISP_TEXT),
    );
    ui.add_space(8.0_f32);

    // -------------------------------------------------------------------------
    // SEZIONE A: Che tipo di registrazione è? (Griglia 2x2)
    // -------------------------------------------------------------------------
    ui.label(
        RichText::new("Che tipo di registrazione è?")
            .size(13.2_f32)
            .strong()
            .color(Color32::from_rgb(214, 224, 238)),
    );
    ui.add_space(4.0_f32);

    let gap = 10.0_f32;
    let half_w = ((total_w - gap) * 0.5_f32).floor();
    let prof_h = 58.0_f32;

    let rows = [
        [SourceProfile::SalaProveLive, SourceProfile::AutoEqIntelligente],
        [SourceProfile::VoceAcustico, SourceProfile::DemoMixPotenziato],
    ];

    for pair in rows {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for prof in pair {
                if draw_source_profile_card(
                    ui,
                    Vec2::new(half_w, prof_h),
                    state.source_profile == prof,
                    prof,
                ) {
                    state.source_profile = prof;
                }
            }
        });
        ui.add_space(4.0_f32);
    }

    ui.add_space(8.0_f32);

    // -------------------------------------------------------------------------
    // SEZIONE B: Quanto vuoi spingere i Bassi? (Bass Boost — 4 card in riga)
    // -------------------------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("Quanto vuoi spingere i Bassi? (Bass Boost)")
                .size(13.2_f32)
                .strong()
                .color(Color32::from_rgb(214, 224, 238)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new("Anti-Clipping Soft-Knee attivo (0% distorsione)")
                    .size(11.2_f32)
                    .color(StudioPalette::EMERALD),
            );
        });
    });
    ui.add_space(4.0_f32);

    let bass_gap = 10.0_f32;
    let bass_w = ((total_w - bass_gap * 3.0_f32) / 4.0_f32).floor();
    let bass_h = 112.0_f32;
    let modes = [
        BassBoostMode::AntiRimbombo,
        BassBoostMode::Bilanciato,
        BassBoostMode::BassBoostPunch,
        BassBoostMode::MegaSubBoost,
    ];

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = bass_gap;
        for mode in modes {
            if draw_bass_boost_card(
                ui,
                Vec2::new(bass_w, bass_h),
                state.bass_mode == mode,
                mode,
            ) {
                state.bass_mode = mode;
            }
        }
    });
}

fn draw_source_profile_card(
    ui: &mut Ui,
    size: Vec2,
    selected: bool,
    profile: SourceProfile,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let accent = profile.accent();
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("prof_hov"), resp.hovered());
        let painter = ui.painter();
        let radius = CornerRadius::same(11);

        let bg = if selected {
            with_alpha(accent, 38)
        } else {
            lerp_color(
                StudioPalette::DEEP_OBSIDIAN,
                StudioPalette::CARD_GLASS,
                hover_t * 0.8_f32,
            )
        };
        let border = if selected {
            brighten(accent, 24)
        } else {
            lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, hover_t * 0.5_f32)
        };

        painter.rect_filled(rect, radius, bg);
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(if selected { 1.6_f32 } else { 1.0_f32 }, border),
            StrokeKind::Inside,
        );

        let left_x = rect.left() + 14.0_f32;
        painter.text(
            Pos2::new(left_x, rect.top() + 11.0_f32),
            Align2::LEFT_TOP,
            profile.label(),
            FontId::proportional(13.8_f32),
            StudioPalette::CRISP_TEXT,
        );
        painter.text(
            Pos2::new(left_x, rect.top() + 31.5_f32),
            Align2::LEFT_TOP,
            profile.subtitle(),
            FontId::proportional(11.6_f32),
            if selected {
                brighten(accent, 38)
            } else {
                StudioPalette::MUTED_LABEL
            },
        );

        let ind_center = Pos2::new(rect.right() - 18.0_f32, rect.center().y);
        if selected {
            painter.circle_filled(ind_center, 9.0_f32, accent);
            painter.text(
                ind_center,
                Align2::CENTER_CENTER,
                "✓",
                FontId::proportional(11.5_f32),
                StudioPalette::CRISP_TEXT,
            );
        } else {
            painter.circle_stroke(
                ind_center,
                7.5_f32,
                Stroke::new(1.1_f32, StudioPalette::SUBTLE_HAIRLINE),
            );
        }
    }

    resp.clicked()
}

fn draw_bass_boost_card(ui: &mut Ui, size: Vec2, selected: bool, mode: BassBoostMode) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let accent = mode.accent();
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("bass_hov"), resp.hovered());
        let sel_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("bass_sel"), selected);
        let painter = ui.painter();
        let radius = CornerRadius::same(12);

        if sel_t > 0.01_f32 {
            let glow_col = if mode == BassBoostMode::BassBoostPunch {
                lerp_color(StudioPalette::WARM_AMBER, StudioPalette::STUDIO_VIOLET, 0.35_f32)
            } else if mode == BassBoostMode::MegaSubBoost {
                lerp_color(StudioPalette::EMERALD, StudioPalette::ELECTRIC_CYAN, 0.35_f32)
            } else {
                accent
            };
            painter.rect_filled(
                rect.expand(2.2_f32),
                CornerRadius::same(14),
                with_alpha(glow_col, (34.0_f32 * sel_t).round() as u8),
            );
        }

        let bg = if selected {
            with_alpha(accent, 38)
        } else {
            lerp_color(
                StudioPalette::DEEP_OBSIDIAN,
                StudioPalette::CARD_GLASS,
                hover_t * 0.85_f32,
            )
        };
        let border = if selected {
            brighten(accent, 28)
        } else {
            lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, hover_t * 0.55_f32)
        };

        painter.rect_filled(rect, radius, bg);
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(if selected { 1.7_f32 } else { 1.0_f32 }, border),
            StrokeKind::Inside,
        );

        // Mini-istogramma luminoso a 4 barre progressive per visualizzare subito la spinta dei bassi
        let lit_bars = mode.bar_level();
        let bar_w = 10.0_f32;
        let bar_gap = 4.5_f32;
        let total_bars_w = 4.0_f32 * bar_w + 3.0_f32 * bar_gap;
        let start_x = rect.center().x - total_bars_w * 0.5_f32;
        let base_y = rect.top() + 34.0_f32;
        let heights = [9.0_f32, 14.0_f32, 19.5_f32, 25.0_f32];

        for (idx, &bh) in heights.iter().enumerate() {
            let bx = start_x + idx as f32 * (bar_w + bar_gap);
            let bar_rect = Rect::from_min_max(Pos2::new(bx, base_y - bh), Pos2::new(bx + bar_w, base_y));
            if idx < lit_bars {
                let bar_col = if selected {
                    brighten(accent, 22)
                } else {
                    with_alpha(accent, 195)
                };
                painter.rect_filled(bar_rect, CornerRadius::same(3), bar_col);
            } else {
                painter.rect_filled(
                    bar_rect,
                    CornerRadius::same(3),
                    Color32::from_rgb(24, 33, 48),
                );
            }
        }

        painter.text(
            Pos2::new(rect.center().x, rect.top() + 45.0_f32),
            Align2::CENTER_TOP,
            mode.label(),
            FontId::proportional(13.6_f32),
            StudioPalette::CRISP_TEXT,
        );
        painter.text(
            Pos2::new(rect.center().x, rect.top() + 65.0_f32),
            Align2::CENTER_TOP,
            mode.subtitle(),
            FontId::proportional(11.8_f32),
            brighten(accent, 28),
        );
        painter.text(
            Pos2::new(rect.center().x, rect.top() + 86.0_f32),
            Align2::CENTER_TOP,
            mode.tech_hint(),
            FontId::monospace(10.0_f32),
            StudioPalette::MUTED_LABEL,
        );
    }

    resp.clicked()
}

fn draw_step_3_vocal_air_master(ui: &mut Ui, state: &mut GuidedWizardState, total_w: f32) {
    ui.label(
        RichText::new("3. Voce, Brillantezza & Spinta Finale")
            .size(18.0_f32)
            .strong()
            .color(StudioPalette::CRISP_TEXT),
    );
    ui.add_space(8.0_f32);

    // 1. Presenza Voce
    draw_three_choice_row(
        ui,
        total_w,
        "Presenza Voce",
        StudioPalette::ELECTRIC_CYAN,
        [
            (
                VocalClarityMode::Naturale.label(),
                state.vocal_clarity == VocalClarityMode::Naturale,
            ),
            (
                VocalClarityMode::PrimoPiano.label(),
                state.vocal_clarity == VocalClarityMode::PrimoPiano,
            ),
            (
                VocalClarityMode::ExtraNitida.label(),
                state.vocal_clarity == VocalClarityMode::ExtraNitida,
            ),
        ],
        |clicked_idx| {
            state.vocal_clarity = match clicked_idx {
                0 => VocalClarityMode::Naturale,
                1 => VocalClarityMode::PrimoPiano,
                _ => VocalClarityMode::ExtraNitida,
            };
        },
    );

    ui.add_space(8.0_f32);

    // 2. Brillantezza Alti
    draw_three_choice_row(
        ui,
        total_w,
        "Brillantezza Alti",
        StudioPalette::STUDIO_VIOLET,
        [
            (
                AirMode::MorbidaCalda.label(),
                state.air_brilliance == AirMode::MorbidaCalda,
            ),
            (
                AirMode::CristallinaStudio.label(),
                state.air_brilliance == AirMode::CristallinaStudio,
            ),
            (
                AirMode::ExtraAria.label(),
                state.air_brilliance == AirMode::ExtraAria,
            ),
        ],
        |clicked_idx| {
            state.air_brilliance = match clicked_idx {
                0 => AirMode::MorbidaCalda,
                1 => AirMode::CristallinaStudio,
                _ => AirMode::ExtraAria,
            };
        },
    );

    ui.add_space(8.0_f32);

    // 3. Volume Master
    draw_three_choice_row(
        ui,
        total_w,
        "Volume Master",
        StudioPalette::EMERALD,
        [
            (
                MasterPushMode::Dinamico.label(),
                state.master_push == MasterPushMode::Dinamico,
            ),
            (
                MasterPushMode::StreamingPotente.label(),
                state.master_push == MasterPushMode::StreamingPotente,
            ),
            (
                MasterPushMode::MassimaSpinta.label(),
                state.master_push == MasterPushMode::MassimaSpinta,
            ),
        ],
        |clicked_idx| {
            state.master_push = match clicked_idx {
                0 => MasterPushMode::Dinamico,
                1 => MasterPushMode::StreamingPotente,
                _ => MasterPushMode::MassimaSpinta,
            };
        },
    );

    ui.add_space(12.0_f32);

    // Pillola di Riepilogo finale
    let summary_accent = if state.split_stems_ai {
        StudioPalette::STUDIO_VIOLET
    } else {
        StudioPalette::EMERALD
    };
    let summary_head = if state.split_stems_ai {
        format!(
            "Verrà avviata la Separazione AI in 4 tracce + {} + {}",
            state.bass_mode.label(),
            state.vocal_clarity.label()
        )
    } else {
        format!(
            "Applicazione istantanea su traccia singola + {} + {}",
            state.bass_mode.label(),
            state.vocal_clarity.label()
        )
    };
    let summary_sub = if state.split_stems_ai {
        "Ascolti subito il brano migliorato; appena pronte le 4 tracce AI, Bass Boost e volumi strumenti si aggiornano da soli."
    } else {
        "Equalizzazione, Bass Boost anti-clipping e Mastering vengono attivati in tempo reale senza attese."
    };

    Frame::NONE
        .fill(with_alpha(summary_accent, 24))
        .corner_radius(CornerRadius::same(11))
        .stroke(Stroke::new(1.0_f32, with_alpha(summary_accent, 115)))
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_width(total_w - 28.0_f32);
            ui.label(
                RichText::new(summary_head)
                    .size(13.0_f32)
                    .strong()
                    .color(StudioPalette::CRISP_TEXT),
            );
            ui.add_space(2.0_f32);
            ui.label(
                RichText::new(summary_sub)
                    .size(11.6_f32)
                    .color(Color32::from_rgb(196, 210, 230)),
            );
        });
}

fn draw_three_choice_row(
    ui: &mut Ui,
    total_w: f32,
    row_title: &str,
    accent: Color32,
    options: [(&str, bool); 3],
    mut on_select: impl FnMut(usize),
) {
    Frame::NONE
        .fill(StudioPalette::DEEP_OBSIDIAN)
        .corner_radius(CornerRadius::same(11))
        .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            let inner_w = total_w - 24.0_f32;
            ui.set_width(inner_w);
            ui.horizontal(|ui| {
                let label_w = 132.0_f32;
                ui.allocate_ui_with_layout(
                    Vec2::new(label_w, 32.0_f32),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.label(
                            RichText::new(row_title)
                                .size(13.0_f32)
                                .strong()
                                .color(Color32::from_rgb(226, 234, 246)),
                        );
                    },
                );

                let gap = 8.0_f32;
                let rem_w = (inner_w - label_w - 8.0_f32).max(240.0_f32);
                let btn_w = ((rem_w - gap * 2.0_f32) / 3.0_f32).floor();

                ui.spacing_mut().item_spacing.x = gap;
                for (idx, (opt_label, is_sel)) in options.into_iter().enumerate() {
                    if draw_segmented_choice_pill(
                        ui,
                        Vec2::new(btn_w, 32.0_f32),
                        opt_label,
                        is_sel,
                        accent,
                    ) {
                        on_select(idx);
                    }
                }
            });
        });
}

fn draw_segmented_choice_pill(
    ui: &mut Ui,
    size: Vec2,
    label: &str,
    selected: bool,
    accent: Color32,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("seg_hov"), resp.hovered());
        let painter = ui.painter();
        let radius = CornerRadius::same(8);

        let bg = if selected {
            accent
        } else {
            lerp_color(
                StudioPalette::ELEVATED_SURFACE,
                StudioPalette::CARD_GLASS,
                hover_t,
            )
        };
        let border = if selected {
            brighten(accent, 32)
        } else {
            lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, hover_t * 0.55_f32)
        };

        if selected {
            painter.rect_filled(
                rect.expand(1.5_f32),
                CornerRadius::same(9),
                with_alpha(accent, 38),
            );
        }

        painter.rect_filled(rect, radius, bg);
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(1.0_f32, border),
            StrokeKind::Inside,
        );

        let text_col = if selected {
            StudioPalette::CRISP_TEXT
        } else {
            lerp_color(
                Color32::from_rgb(190, 204, 224),
                StudioPalette::CRISP_TEXT,
                hover_t,
            )
        };
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(12.2_f32),
            text_col,
        );
    }

    resp.clicked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::StudioDspChain;

    #[test]
    fn wizard_defaults_and_stem_offsets_match_spec() {
        let mut st = GuidedWizardState::default();
        st.open_for_track("prova_band.mp3");
        assert!(st.open);
        assert_eq!(st.step, 1);
        assert!(st.split_stems_ai);
        assert_eq!(st.bass_mode, BassBoostMode::BassBoostPunch);

        let outcome = st.build_outcome(None);
        assert!(outcome.run_ai_split);
        assert!((outcome.stem_gains_db[0] - 1.8).abs() < 1e-4);
        assert!((outcome.stem_gains_db[1] - 1.5).abs() < 1e-4);
        assert!((outcome.stem_gains_db[2] - 3.5).abs() < 1e-4);
    }

    #[test]
    fn mega_sub_boost_guarantees_zero_clipping() {
        let choices = GuidedWizardChoices {
            split_stems_ai: false,
            source_profile: SourceProfile::DemoMixPotenziato,
            bass_mode: BassBoostMode::MegaSubBoost,
            vocal_clarity: VocalClarityMode::ExtraNitida,
            air_brilliance: AirMode::ExtraAria,
            master_push: MasterPushMode::MassimaSpinta,
        };
        let outcome = choices.build_outcome(None);
        let mut chain = StudioDspChain::new(48_000.0, &outcome.params);
        let ceiling_lin = 10.0_f32.powf((outcome.params.ceiling_db as f32) / 20.0) + 1e-4;

        for n in 0..4096 {
            let t = n as f32 / 48_000.0;
            let sub = (2.0 * std::f32::consts::PI * 55.0 * t).sin() * 0.95;
            let kick = (2.0 * std::f32::consts::PI * 90.0 * t).sin() * 0.65;
            let (l, r) = chain.process_frame(sub + kick, sub + kick, &outcome.params);
            assert!(l.is_finite() && r.is_finite());
            assert!(l.abs() <= ceiling_lin && r.abs() <= ceiling_lin);
        }
    }
}
