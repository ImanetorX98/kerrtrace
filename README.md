# KerrTrace (Python + GPU)

Progetto Python per ray tracing gravitazionale di un buco nero rotante (metrica di Kerr) con disco di accrescimento sottile, vettorizzato su GPU tramite PyTorch.

## Cosa fa

- Integra geodetiche nulle in coordinate Boyer-Lindquist, Kerr-Schild e Generalized-Doran/PG-like con schema RK45 adattivo (fallback RK4 opzionale).
- Supporta metrica selezionabile: `schwarzschild`, `kerr`, `reissner_nordstrom`, `kerr_newman` e varianti `_de_sitter`.
- Usa orientazione camera con quaternioni (riduce artefatti da singolarita' di angolo in animazione/orbita).
- Trova l'intersezione dei raggi con il disco equatoriale tramite raffinamento evento (interpolazione Hermite).
- Supporta raggio interno/esterno del disco con boost separato dei bordi (look stile Interstellar).
- Supporta disco fisico (profilo termico tipo Novikov-Thorne + blackbody) e fallback palette plasma.
- Applica redshift relativistico e boost Doppler locale per il colore/intensita' osservata.
- Genera raggi camera con tetrade locale fisica (ZAMO).
- Esegue il calcolo in batch su `cuda` (se disponibile) o fallback `cpu`.
- Include sfondo galattico procedurale multi-layer o HDRI equirettangolare.
- Supporta animazioni camera-orbit con TAA/motion blur ed export video (`.mp4/.mov/.mkv`) o GIF.

## Requisiti

- Python 3.10+
- GPU NVIDIA con CUDA (opzionale ma consigliata)
- `ffmpeg` nel PATH per export video MP4/MOV/MKV (non serve per GIF)

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Oppure installazione package locale:

```bash
pip install -e .
```

## Web UI interattiva

Puoi lanciare simulazioni frame/video da browser locale, con parametri interattivi e preset qualita':
- `144p`, `288p`,`480p`, `720p`, `1080p`, `2K`, `4K`
- modalita' `Single Frame` o `Video`
- editor JSON avanzato per sovrascrivere qualsiasi campo del `RenderConfig`
- profili performance (`Manual`, `GPU Balanced`, `Fast Preview`, `High Fidelity`)
- controllo `render_tile_rows` per tuning throughput/progress
- toggle `camera_fastpath` (legacy vs ottimizzato)
- avvio live o in background con monitor log/output
- esecuzione tramite `python -m kerrtrace --config ...` con log live

Avvio:

```bash
streamlit run kerrtrace/webui.py
```

Nota: la UI genera un file JSON di configurazione in `out/webui_runs/` e usa quello per il run.

## Esecuzione rapida

```bash
python -m kerrtrace --width 960 --height 540 --device auto --output out/render.png
```

Benchmark tempi con report JSON+Markdown:

```bash
.venv/bin/python scripts/benchmark_render_speed.py --device mps --width 854 --height 480 --repeats 2 --warmup 1 --scenario both --require-gpu
```

Con file JSON (input complessi):

```bash
python -m kerrtrace --config example_config.json
```

Sovrascrivendo alcuni parametri da CLI:

```bash
python -m kerrtrace --config example_config.json --spin 0.98 --max-steps 1800 --output out/spin098.png
```

Esempio metrica Kerr-Newman-de Sitter:

```bash
python -m kerrtrace --config example_config.json --metric-model kerr_newman_de_sitter --spin 0.9 --charge 0.2 --cosmological-constant 0.002 --output out/knds.png
```

Preset piu' cinematografico:

```bash
python -m kerrtrace --config example_config.json --inner-edge-boost 2.8 --outer-edge-boost 0.5 --star-density 0.0025 --output out/interstellar_style.png
```

## Animazioni e video

Orbit camera e salva MP4:

```bash
python -m kerrtrace --config example_config.json --animate --frames 120 --fps 30 --azimuth-orbits 1.0 --output out/orbit.mp4
```

Orbit con wobble verticale e salva GIF:

```bash
python -m kerrtrace --config example_config.json --animate --frames 90 --fps 20 --azimuth-orbits 1.2 --inclination-wobble-deg 3.0 --output out/orbit.gif
```

Passaggio lento da sopra a sotto il piano del disco:

```bash
python -m kerrtrace --config example_config.json --animate --frames 96 --fps 12 --azimuth-orbits 0.35 --inclination-start-deg 55 --inclination-end-deg 125 --output out/above_to_below.mp4
```

Sweep di osservatore in caduta (raggio variabile, attraversamento orizzonte in coordinate PG-like):

```bash
python -m kerrtrace --config example_config.json --coordinate-system generalized_doran --metric-model kerr_newman_de_sitter --animate --frames 120 --fps 24 --observer-radius-start 30 --observer-radius-end 1.5 --inclination-start-deg 45 --inclination-end-deg 135 --output out/infall_gdoran.mp4
```

Sweep in caduta con campionamento a tempo uniforme (inversione numerica di `t(r)` in modalità generalized Doran):

```bash
python -m kerrtrace --config example_config.json --coordinate-system generalized_doran --animate --frames 120 --fps 24 --observer-radius-start 40 --observer-radius-end 10 --generalized-doran-fixed-time --generalized-doran-radius-log out/infall_schedule.csv --output out/infall_fixed_dt.mp4
```

Mantieni anche i frame PNG:

```bash
python -m kerrtrace --config example_config.json --animate --frames 60 --fps 24 --output out/shot.mp4 --keep-frames --frames-dir out/shot_frames
```

Diagnostica device (utile su Mac per MPS):

```bash
python -m kerrtrace --config example_config.json --diagnose-device
```

Se vuoi evitare fallback silenzioso su CPU:

```bash
python -m kerrtrace --config example_config.json --device mps --require-gpu --output out/render_mps.png
```

Se MPS e' disponibile, forza GPU Apple:

```bash
python -m kerrtrace --config example_config.json --device mps --output out/render_mps.png
```

Attiva il kernel ottimizzato per MPS (fast path):

```bash
python -m kerrtrace --config example_config.json --device mps --mps-optimized-kernel --output out/render_mps_fast.png
```

Per ridurre artefatti di seam del meridiano:

```bash
python -m kerrtrace --config example_config.json --enable-meridian-destripe --background-meridian-offset-deg 137.5 --output out/render_no_seam.png
```

Fix hard della seam via cubemap:

```bash
python -m kerrtrace --config example_config.json --background-projection cubemap --cubemap-face-size 768 --output out/render_cubemap.png
```

Nota Codex Desktop: nelle esecuzioni sandbox (`CODEX_SANDBOX=seatbelt`) MPS puo' risultare non disponibile anche su Mac compatibili. Esegui il comando in un terminale arm64 nativo fuori sandbox per usare la GPU.

Setup rapido ambiente MPS (Python 3.11):

```bash
./scripts/setup_mps_env.sh /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 .venv-mps
```

## Parametri principali

- `spin`: parametro adimensionale del buco nero (`|a| < 1`).
- `metric_model`: scelta metrica (`schwarzschild`, `kerr`, `reissner_nordstrom`, `kerr_newman`, e versioni `_de_sitter`).
- `charge`: parametro di carica (usato nelle metriche Reissner-Nordstrom/Kerr-Newman).
- `cosmological_constant`: costante cosmologica `Lambda` (usata nelle metriche `_de_sitter`).
  - Nota: per `Lambda > 0` l'osservatore deve stare dentro l'orizzonte cosmologico.
- `observer_radius`: distanza osservatore in unita' di massa `M`.
- `observer_inclination_deg`: inclinazione osservatore (0=asse, 90=piano disco).
- `observer_azimuth_deg`: rotazione azimutale iniziale della camera.
- `observer_roll_deg`: roll camera (applicato via quaternioni).
- `coordinate_system`: `boyer_lindquist`, `kerr_schild`, `generalized_doran`.
  - `generalized_doran`: frame osservatore PG-like/infalling su base GKS, utile per attraversare l'orizzonte.
- `disk_inner_radius`: raggio interno del disco. Se `null`, usa ISCO prograde.
- `disk_outer_radius`: raggio esterno del disco.
- `inner_edge_boost`, `outer_edge_boost`: luminosita' extra sui bordi interno/esterno del disco.
- `step_size`, `max_steps`: controllo accuratezza/costo integrazione.
- `adaptive_integrator`: usa integrazione RK45 adattiva (default `true`).
- `adaptive_rtol`, `adaptive_atol`: tolleranze relative/assolute del controllo errore locale.
- `adaptive_step_min`, `adaptive_step_max`: limiti del passo adattivo.
- `device`: `auto`, `cpu`, `cuda`, `mps` (GPU Apple Silicon via Metal).
- `--require-gpu`: fallisce se il rendering andrebbe su CPU (utile per evitare fallback involontario).
- `dtype`: `float32` (veloce), `float64` (piu' stabile ma piu' lento).
- `enable_star_background`: attiva/disattiva lo sfondo di stelle.
- `background_mode`: `procedural` o `hdri`.
- `background_projection`: `cubemap` (consigliato, seam-free) o `equirectangular` (legacy).
- `cubemap_face_size`: risoluzione per faccia della cubemap (tipico: 512-1024).
- `hdri_path`, `hdri_exposure`, `hdri_rotation_deg`: controlli sfondo HDRI.
- `background_meridian_offset_deg`: ruota il meridiano di seam dello sfondo.
- `star_density`, `star_brightness`, `star_seed`: densita', intensita' e seed dello sfondo stellare.
- `disk_model`: `physical_nt` (profilo fisico thin-disk relativistico) oppure `legacy` (palette cinematica).
- `disk_temperature_inner`: temperatura caratteristica del bordo interno (K).
- `disk_color_correction`: hardening factor spettrale (tipico 1.5-1.9) per il modello `physical_nt`.
- `disk_plasma_warmth`: mix [0..1] verso tinta plasma calda nel modello `physical_nt`.
- `thick_disk`: abilita il modello di disco spesso (default `false`, disco sottile).
- `disk_thickness_ratio`: semispessore relativo `H/r` (tipico 0.08-0.18).
- `disk_thickness_power`: legge radiale dello spessore `H(r) ~ r^(1+power)`.
- `physical_disk_model`: alias legacy (se `false`, forza `disk_model=legacy`).
- `enforce_black_hole_shadow`, `shadow_absorb_radius_factor`: soppressione luce spuria dentro l'ombra del buco nero.
- `destripe_meridian`: hook legacy (default `false`).
- `--enable-meridian-destripe`: applica smoothing locale automatico sulla colonna di seam.
- `animate`, `frames`, `fps`, `azimuth_orbits`, `inclination_wobble_deg`: controlli animazione camera.
- `inclination_start_deg`, `inclination_end_deg`: sweep dell'inclinazione da un valore iniziale a uno finale.
- `observer_radius_start`, `observer_radius_end`: sweep lineare del raggio osservatore durante l'animazione (utile per traiettorie di caduta).
- `generalized_doran_fixed_time`: in `generalized_doran`, sostituisce lo sweep lineare con frame a passo temporale uniforme, invertendo numericamente `t(r)`.
- `generalized_doran_time_samples`: numero di campioni usati per integrare/invertire `t(r)` (piu' alto = piu' stabile ma piu' lento).
- `generalized_doran_radius_log`: CSV opzionale con mappa `frame,time,radius` della traiettoria usata.
- `taa_samples`, `shutter_fraction`, `spatial_jitter`: anti-aliasing temporale e motion blur.
- `postprocess_pipeline`: `off` o `gargantua` per grading cinematografico in post.
- `gargantua_look_strength`: intensita' [0..2] del look `gargantua`.
- `compile_rhs`, `mixed_precision`: opzioni performance GPU.
- `mps_optimized_kernel`: abilita fast path MPS (integrazione fixed-step con interpolazione eventi leggera, piu' veloce ma meno accurata del percorso completo adattivo).
- `camera_fastpath`: abilita init raggi camera ottimizzata (disabilitabile per confronti benchmark con percorso legacy).
- `diagnose_device`: stampa diagnostica backend (`cuda`/`mps`) e suggerimenti.

## Note numeriche

- Questa e' una reference implementation orientata a prototipazione visuale.
- Per risultati scientifici ad alta fedelta': usare integrazione adattiva, tetradi complete per l'osservatore, test di convergenza e validazione contro benchmark GRRT.

## Output

Il renderer salva un PNG e stampa statistiche:

- numero totale di raggi
- quanti colpiscono il disco
- quanti attraversano l'orizzonte
- quanti sfuggono
- numero di step usati
