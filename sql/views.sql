-- Analysis views for the nuclear scaling database.
-- Install once per DB file (nsdb.install_views). Re-running is safe.
--
-- Design rule: raw tables are written by the import pipeline and never by
-- analysis code. Views are the only thing analysis reads. If an analysis needs
-- a new derived column, it goes here -- not into a notebook cell.

DROP VIEW IF EXISTS v_nuclei;
DROP VIEW IF EXISTS v_nuclei_intensity;
DROP VIEW IF EXISTS v_intensity_long;
DROP VIEW IF EXISTS v_zstack;


-- ---------------------------------------------------------------------------
-- v_nuclei : the workhorse. One row per (nucleus, time_frame), with all
-- experimental metadata denormalised onto it. This is what you group by when
-- you add replicates and conditions.
-- ---------------------------------------------------------------------------
CREATE VIEW v_nuclei AS
SELECT
    -- identity
    n.nucleus_id,
    n.time_frame,
    -- Two different clocks, deliberately both exposed.
    -- time_min: true acquisition time, varying within a frame because the
    --   mosaic is scanned tile by tile. Use this for anything kinetic.
    -- frame_time_min: the nominal frame label. Use for grouping/faceting only.
    n.time_real_minutes AS time_min,
    n.time_frame * c.frame_interval_min AS frame_time_min,
    n.time_real_minutes - (n.time_frame * c.frame_interval_min) AS tile_offset_min,
    n.tile_index,
    n.tile_row,
    n.tile_col,
    n.nucleus_track_id,
    n.source_detection_id,

    -- grouping keys: these are what make added replicates "just work"
    n.experiment_id,
    n.fov_id,
    n.droplet_id,
    c.experimental_group,
    c.treatment,
    c.concentration,
    c.concentration_units,
    c.biological_replicate,
    c.technical_replicate,
    c.extract_batch,
    c.experiment_date,

    -- measurements
    n.cross_sectional_area_um2,
    n.volume_3d_um3,
    n.nc_ratio,
    n.centroid_x_px,
    n.centroid_y_px,
    n.centroid_z_px,
    n.selected_slice_id,
    n.stage_classification,

    -- provenance / QC
    n.qc_flag,
    n.qc_notes,
    n.droplet_assignment_status,
    n.zstack_consistency,
    n.repair_status,
    n.repair_gain,
    n.ratio_vs_npc,
    n.segmentation_pipeline_version,
    n.model_version,

    -- acquisition parameters, for unit conversions done at analysis time
    c.pixel_size_um,
    c.z_step_um,
    c.num_z_planes,
    c.frame_interval_min,
    c.channel0_label,
    c.channel1_label,
    c.channel2_label
FROM nuclei n
JOIN experimental_cfg c
  ON  n.experiment_id = c.experiment_id
  AND n.fov_id        = c.fov_id
  AND n.droplet_id    = c.droplet_id;


-- ---------------------------------------------------------------------------
-- v_nuclei_intensity : v_nuclei plus the wide halo intensity columns.
-- LEFT JOIN so a nucleus with no intensity row still appears (and shows up as
-- NULL rather than silently vanishing from your n).
-- ---------------------------------------------------------------------------
CREATE VIEW v_nuclei_intensity AS
SELECT
    v.*,
    r.halo1_mcherry,  r.halo2_mcherry,  r.halo3_mcherry,  r.halo4_mcherry,
    r.halo1_npc,      r.halo2_npc,      r.halo3_npc,      r.halo4_npc,
    r.halo1_membrane, r.halo2_membrane, r.halo3_membrane, r.halo4_membrane,
    r.background_mcherry,
    r.background_npc,
    r.background_membrane
FROM v_nuclei v
LEFT JOIN raw_intensities r
       ON  v.nucleus_id = r.nucleus_id
       AND v.time_frame = r.time_frame;


-- ---------------------------------------------------------------------------
-- v_intensity_long : tidy form -- one row per (nucleus, frame, halo, channel).
-- This is the shape seaborn and ggplot2 both want. Background is carried
-- alongside so background-subtracted values are a single arithmetic step.
-- ---------------------------------------------------------------------------
CREATE VIEW v_intensity_long AS
SELECT nucleus_id, time_frame, 1 AS halo, 'mcherry'  AS channel,
       halo1_mcherry  AS intensity, background_mcherry  AS background FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 2, 'mcherry',  halo2_mcherry,  background_mcherry  FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 3, 'mcherry',  halo3_mcherry,  background_mcherry  FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 4, 'mcherry',  halo4_mcherry,  background_mcherry  FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 1, 'npc',      halo1_npc,      background_npc      FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 2, 'npc',      halo2_npc,      background_npc      FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 3, 'npc',      halo3_npc,      background_npc      FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 4, 'npc',      halo4_npc,      background_npc      FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 1, 'membrane', halo1_membrane, background_membrane FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 2, 'membrane', halo2_membrane, background_membrane FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 3, 'membrane', halo3_membrane, background_membrane FROM raw_intensities
UNION ALL SELECT nucleus_id, time_frame, 4, 'membrane', halo4_membrane, background_membrane FROM raw_intensities;


-- ---------------------------------------------------------------------------
-- v_zstack : per-plane rows with metadata, for z-profile and cap-artifact work.
-- is_selected_max marks the plane that fed the nuclei-table measurement.
-- ---------------------------------------------------------------------------
CREATE VIEW v_zstack AS
SELECT
    z.nucleus_id,
    z.time_frame,
    z.slice_id,
    z.centroid_x_px,
    z.centroid_y_px,
    z.centroid_z_px,
    z.cross_sectional_area_um2,
    z.is_selected_max,
    z.repair_source,
    v.experiment_id,
    v.fov_id,
    v.droplet_id,
    v.experimental_group,
    v.treatment,
    v.biological_replicate,
    v.qc_flag,
    v.pixel_size_um,
    v.z_step_um
FROM nucleus_z_stack z
JOIN v_nuclei v
  ON  z.nucleus_id = v.nucleus_id
  AND z.time_frame = v.time_frame;
