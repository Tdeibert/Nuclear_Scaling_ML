-- ===========================================================================
-- Nuclear scaling database -- schema v2
--
-- Changes from the schema in nuclear_scaling_test.db, and why:
--
-- 1. Condition columns are NOT NULL. experimental_group, treatment,
--    biological_replicate and extract_batch cannot be omitted. This is the
--    single most important change: it makes the "every condition column is
--    NULL" failure structurally impossible rather than merely discouraged.
--
-- 2. time_real_minutes is KEPT, and is the primary time axis. It varies
--    within a time_frame because the FOV is a stitched mosaic acquired tile
--    by tile: add_tile_timing_metadata computes
--        true_time_min = t * tiles_per_frame + tile_index * minutes_per_tile
--    so nuclei in different tiles of the same frame are genuinely imaged up
--    to (tiles-1) * minutes_per_tile apart. On control_extract_1.1 that is a
--    3x2 serpentine grid at 1 min/tile: offsets 0,1,2 across the top row and
--    5,4,3 across the bottom. Use this for kinetics. The nominal
--    time_frame * frame_interval_min is exposed separately as frame_time_min
--    and is only a frame label.
--
-- 3. experiment_id must carry a replicate suffix (GLOB backstop). Real
--    validation is in experiment_config.py; this catches anything that
--    bypasses it.
--
-- 4. New table import_issues. The audit result is recorded at import time and
--    kept, so months later you can tell whether a given run was clean without
--    re-deriving it.
--
-- 5. New table rejected_rows. Rows that fail structural validation are
--    quarantined with a reason rather than silently dropped or silently
--    admitted.
--
-- 6. nuclei.selected_slice_id deliberately has NO foreign key into
--    nucleus_z_stack. In the reference export 81 nuclei point at a plane
--    absent from the z-stack, because the v17 fragmentation repair re-selects
--    z (measuring planes that produced no original detection) and overwrites
--    best_z_df without writing the new planes back into grouped_z_df, which
--    is what NucleusZStack is built from. The nuclei row holds the repaired,
--    more accurate measurement; the z-stack is the stale table. Enforcing the
--    FK would discard the better data to satisfy the worse. The mismatch is
--    reported by nsdb.audit() instead.
-- ===========================================================================

PRAGMA foreign_keys = ON;


-- ---------------------------------------------------------------------------
CREATE TABLE experimental_cfg (
    experiment_id   TEXT NOT NULL
        CHECK (experiment_id GLOB '*.[0-9]*'),   -- must end in a replicate index
    fov_id          TEXT NOT NULL,
    droplet_id      TEXT NOT NULL DEFAULT 'UNASSIGNED',

    -- condition: required, because analysis groups on these
    experimental_group   TEXT NOT NULL,
    treatment            TEXT NOT NULL,
    concentration        REAL,
    concentration_units  TEXT,
    biological_replicate TEXT NOT NULL,
    technical_replicate  TEXT,
    extract_batch        TEXT NOT NULL,

    -- acquisition
    experiment_date     TEXT NOT NULL,
    pixel_size_um       REAL NOT NULL CHECK (pixel_size_um > 0),
    z_step_um           REAL NOT NULL CHECK (z_step_um > 0),
    num_z_planes        INTEGER NOT NULL CHECK (num_z_planes > 0),
    frame_interval_min  REAL NOT NULL CHECK (frame_interval_min > 0),

    -- Mosaic scan geometry. The FOV is stitched from tiles acquired one after
    -- another, so nuclei in different tiles of the same frame are imaged
    -- minutes apart. These make tile_index recoverable from time_real_minutes
    -- and must satisfy tile_rows * tile_cols * minutes_per_tile =
    -- frame_interval_min (checked in experiment_config.py).
    tile_rows       INTEGER CHECK (tile_rows > 0),
    tile_cols       INTEGER CHECK (tile_cols > 0),
    minutes_per_tile REAL   CHECK (minutes_per_tile > 0),
    serpentine_scan INTEGER CHECK (serpentine_scan IN (0, 1)),
    channel0_label      TEXT NOT NULL DEFAULT 'Membrane',
    channel1_label      TEXT NOT NULL DEFAULT 'NLS',
    channel2_label      TEXT NOT NULL DEFAULT 'NPC',
    microscope          TEXT,
    objective           TEXT,
    operator            TEXT NOT NULL,

    -- provenance
    segmentation_pipeline_version TEXT,
    model_version                 TEXT,
    raw_file_path                 TEXT,
    notes                         TEXT,

    PRIMARY KEY (experiment_id, fov_id, droplet_id),

    -- a named compound means a concentration is given; 'none' means zero.
    -- Table-level, so it must come after every column definition.
    CHECK (
        (lower(treatment) IN ('none','untreated','vehicle','buffer')
         AND (concentration IS NULL OR concentration = 0))
        OR (concentration IS NOT NULL AND concentration_units IS NOT NULL)
    )
);


-- ---------------------------------------------------------------------------
CREATE TABLE nuclei (
    nucleus_id    TEXT NOT NULL,
    time_frame    INTEGER NOT NULL CHECK (time_frame >= 0),

    droplet_id    TEXT NOT NULL,
    fov_id        TEXT NOT NULL,
    experiment_id TEXT NOT NULL,

    -- True per-tile acquisition time. NOT derivable from time_frame; see
    -- note 2 in the header. This is the time axis for kinetics.
    time_real_minutes    REAL CHECK (time_real_minutes >= 0),

    -- Derived at import by inverting time_real_minutes against the tile grid.
    -- Lets you treat scan position as a nuisance variable, or check for
    -- systematic differences between tiles.
    tile_index INTEGER CHECK (tile_index >= 0),
    tile_row   INTEGER CHECK (tile_row >= 0),
    tile_col   INTEGER CHECK (tile_col >= 0),

    stage_classification TEXT,
    selected_slice_id    TEXT NOT NULL,
    centroid_x_px        REAL,
    centroid_y_px        REAL,
    centroid_z_px        REAL,

    cross_sectional_area_um2 REAL CHECK (cross_sectional_area_um2 >= 0),
    volume_3d_um3            REAL CHECK (volume_3d_um3 >= 0),
    nc_ratio                 REAL CHECK (nc_ratio >= 0),

    segmentation_pipeline_version TEXT,
    model_version                 TEXT,
    qc_flag TEXT NOT NULL DEFAULT 'UNREVIEWED'
        CHECK (qc_flag IN ('PASS','FAIL','REVIEW','UNREVIEWED')),
    qc_notes                  TEXT,
    source_file_path          TEXT,
    source_detection_id       TEXT,
    nucleus_track_id          TEXT,
    droplet_assignment_status TEXT NOT NULL DEFAULT 'UNASSIGNED'
        CHECK (droplet_assignment_status IN ('ASSIGNED','UNASSIGNED')),

    -- How this row's measurement relates to nucleus_z_stack. Derived at
    -- import; see the header note on the v17 repair.
    --   consistent     -- selected plane present, areas agree
    --   area_mismatch  -- selected plane present, areas differ
    --   plane_missing  -- selected plane absent from the z-stack entirely
    zstack_consistency TEXT
        CHECK (zstack_consistency IN ('consistent','area_mismatch','plane_missing')),

    -- Populated once the export carries them (pipeline v18.2+). NULL means
    -- the export predates repair provenance, not that no repair happened.
    repair_status TEXT,
    repair_gain   REAL CHECK (repair_gain > 0),
    ratio_vs_npc  REAL CHECK (ratio_vs_npc > 0),

    import_run_id INTEGER,

    PRIMARY KEY (nucleus_id, time_frame),
    FOREIGN KEY (experiment_id, fov_id, droplet_id)
        REFERENCES experimental_cfg (experiment_id, fov_id, droplet_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);


-- ---------------------------------------------------------------------------
CREATE TABLE nucleus_z_stack (
    nucleus_id   TEXT NOT NULL,
    time_frame   INTEGER NOT NULL CHECK (time_frame >= 0),
    slice_id     TEXT NOT NULL,
    centroid_x_px REAL,
    centroid_y_px REAL,
    centroid_z_px REAL,
    cross_sectional_area_um2 REAL CHECK (cross_sectional_area_um2 >= 0),
    is_selected_max INTEGER NOT NULL CHECK (is_selected_max IN (0,1)),

    -- Whether this plane's measurement reflects the fragmentation repair.
    --   original            -- as detected by the model
    --   repaired            -- existing plane, remeasured after repair
    --   repaired_new_plane  -- plane the repair measured for the first time
    --   unrepaired_stale    -- export predates the v18.2 write-back, so this
    --                          row is pre-repair even if the nucleus was
    --                          repaired. Do NOT use for area measurements.
    repair_source TEXT NOT NULL DEFAULT 'original'
        CHECK (repair_source IN ('original','repaired','repaired_new_plane',
                                 'unrepaired_stale')),
    import_run_id INTEGER,
    PRIMARY KEY (nucleus_id, time_frame, slice_id)
);

CREATE UNIQUE INDEX ux_one_selected_slice
    ON nucleus_z_stack (nucleus_id, time_frame) WHERE is_selected_max = 1;


-- ---------------------------------------------------------------------------
CREATE TABLE raw_intensities (
    nucleus_id TEXT NOT NULL,
    time_frame INTEGER NOT NULL CHECK (time_frame >= 0),
    halo1_mcherry REAL, halo2_mcherry REAL, halo3_mcherry REAL, halo4_mcherry REAL,
    halo1_npc REAL, halo2_npc REAL, halo3_npc REAL, halo4_npc REAL,
    halo1_membrane REAL, halo2_membrane REAL, halo3_membrane REAL, halo4_membrane REAL,
    background_mcherry REAL, background_npc REAL, background_membrane REAL,
    PRIMARY KEY (nucleus_id, time_frame),
    FOREIGN KEY (nucleus_id, time_frame) REFERENCES nuclei (nucleus_id, time_frame)
        ON UPDATE CASCADE ON DELETE CASCADE
);


-- ---------------------------------------------------------------------------
-- Bulk radial rows live in parquet; only the pointer is stored here.
CREATE TABLE radial_profile_files (
    radial_file_id INTEGER PRIMARY KEY,
    experiment_id  TEXT NOT NULL,
    fov_id         TEXT NOT NULL,
    droplet_id     TEXT NOT NULL,
    source_csv_path    TEXT NOT NULL,
    parquet_path       TEXT NOT NULL,
    row_count          INTEGER NOT NULL CHECK (row_count >= 0),
    parquet_size_bytes INTEGER NOT NULL CHECK (parquet_size_bytes >= 0),
    compression   TEXT NOT NULL DEFAULT 'zstd',
    created_at    TEXT NOT NULL,
    import_run_id INTEGER,
    UNIQUE (experiment_id, fov_id, droplet_id, parquet_path),
    FOREIGN KEY (experiment_id, fov_id, droplet_id)
        REFERENCES experimental_cfg (experiment_id, fov_id, droplet_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);


-- ---------------------------------------------------------------------------
CREATE TABLE import_runs (
    import_run_id INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    source_directory TEXT NOT NULL,
    config_path      TEXT,
    experiment_id TEXT,
    fov_id        TEXT,
    status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
    nuclei_rows        INTEGER,
    z_stack_rows       INTEGER,
    raw_intensity_rows INTEGER,
    radial_rows        INTEGER,
    rejected_rows      INTEGER,
    schema_version     TEXT NOT NULL DEFAULT 'v2',
    message TEXT
);


-- Audit results, recorded per import and kept.
CREATE TABLE import_issues (
    issue_id      INTEGER PRIMARY KEY,
    import_run_id INTEGER NOT NULL REFERENCES import_runs (import_run_id) ON DELETE CASCADE,
    severity   TEXT NOT NULL CHECK (severity IN ('INFO','WARN','ERROR')),
    check_name TEXT NOT NULL,
    n          INTEGER NOT NULL,
    detail     TEXT
);


-- Rows that failed structural validation, kept with a reason rather than
-- dropped. Payload is the original CSV row as JSON.
CREATE TABLE rejected_rows (
    rejected_id   INTEGER PRIMARY KEY,
    import_run_id INTEGER NOT NULL REFERENCES import_runs (import_run_id) ON DELETE CASCADE,
    source_table  TEXT NOT NULL,
    reason        TEXT NOT NULL,
    payload       TEXT NOT NULL
);


-- ---------------------------------------------------------------------------
CREATE INDEX ix_nuclei_experiment ON nuclei (experiment_id, fov_id, droplet_id);
CREATE INDEX ix_nuclei_qc         ON nuclei (qc_flag);
CREATE INDEX ix_nuclei_frame      ON nuclei (experiment_id, time_frame);
CREATE INDEX ix_zstack_selected   ON nucleus_z_stack (nucleus_id, time_frame, is_selected_max);
CREATE INDEX ix_issues_run        ON import_issues (import_run_id, severity);
CREATE INDEX ix_radial_experiment ON radial_profile_files (experiment_id, fov_id);


-- ---------------------------------------------------------------------------
-- Keeps nuclei.selected_slice_id and nucleus_z_stack.is_selected_max agreeing.
CREATE TRIGGER validate_selected_slice_insert
BEFORE INSERT ON nucleus_z_stack
WHEN NEW.is_selected_max = 1
BEGIN
    SELECT CASE WHEN NEW.slice_id != (
        SELECT selected_slice_id FROM nuclei
        WHERE nucleus_id = NEW.nucleus_id AND time_frame = NEW.time_frame)
    THEN RAISE(ABORT, 'is_selected_max slice does not match nuclei.selected_slice_id')
    END;
END;

CREATE TRIGGER validate_selected_slice_update
BEFORE UPDATE OF is_selected_max, slice_id ON nucleus_z_stack
WHEN NEW.is_selected_max = 1
BEGIN
    SELECT CASE WHEN NEW.slice_id != (
        SELECT selected_slice_id FROM nuclei
        WHERE nucleus_id = NEW.nucleus_id AND time_frame = NEW.time_frame)
    THEN RAISE(ABORT, 'is_selected_max slice does not match nuclei.selected_slice_id')
    END;
END;
