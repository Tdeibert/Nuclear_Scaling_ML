PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experimental_cfg (
    experiment_id TEXT NOT NULL,
    fov_id TEXT NOT NULL,
    droplet_id TEXT NOT NULL,
    experiment_date TEXT,
    pixel_size_um REAL NOT NULL CHECK (pixel_size_um > 0),
    z_step_um REAL NOT NULL CHECK (z_step_um > 0),
    num_z_planes INTEGER NOT NULL CHECK (num_z_planes > 0),
    frame_interval_min REAL NOT NULL CHECK (frame_interval_min > 0),
    channel0_label TEXT,
    channel1_label TEXT,
    channel2_label TEXT,
    microscope TEXT,
    objective TEXT,
    operator TEXT,
    segmentation_pipeline_version TEXT,
    model_version TEXT,
    raw_file_path TEXT,
    treatment TEXT,
    concentration REAL,
    concentration_units TEXT,
    biological_replicate TEXT,
    technical_replicate TEXT,
    extract_batch TEXT,
    experimental_group TEXT,
    notes TEXT,
    PRIMARY KEY (experiment_id, fov_id, droplet_id)
);

CREATE TABLE IF NOT EXISTS nuclei (
    nucleus_id TEXT NOT NULL,
    droplet_id TEXT NOT NULL,
    fov_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    time_frame INTEGER NOT NULL CHECK (time_frame >= 0),
    time_real_minutes REAL CHECK (time_real_minutes >= 0),
    stage_classification TEXT,
    selected_slice_id TEXT,
    centroid_x_px REAL,
    centroid_y_px REAL,
    centroid_z_px REAL,
    cross_sectional_area_um2 REAL CHECK (cross_sectional_area_um2 >= 0),
    volume_3d_um3 REAL CHECK (volume_3d_um3 >= 0),
    nc_ratio REAL,
    segmentation_pipeline_version TEXT,
    model_version TEXT,
    qc_flag TEXT NOT NULL DEFAULT 'UNREVIEWED'
        CHECK (qc_flag IN ('PASS', 'FAIL', 'REVIEW', 'UNREVIEWED')),
    qc_notes TEXT,
    source_file_path TEXT,
    PRIMARY KEY (nucleus_id, time_frame),
    FOREIGN KEY (experiment_id, fov_id, droplet_id)
        REFERENCES experimental_cfg (experiment_id, fov_id, droplet_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS nucleus_z_stack (
    nucleus_id TEXT NOT NULL,
    time_frame INTEGER NOT NULL CHECK (time_frame >= 0),
    slice_id TEXT NOT NULL,
    centroid_x_px REAL,
    centroid_y_px REAL,
    centroid_z_px REAL,
    cross_sectional_area_um2 REAL CHECK (cross_sectional_area_um2 >= 0),
    is_selected_max INTEGER NOT NULL CHECK (is_selected_max IN (0, 1)),
    PRIMARY KEY (nucleus_id, time_frame, slice_id),
    FOREIGN KEY (nucleus_id, time_frame)
        REFERENCES nuclei (nucleus_id, time_frame)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_one_selected_slice
ON nucleus_z_stack (nucleus_id, time_frame)
WHERE is_selected_max = 1;

CREATE TABLE IF NOT EXISTS raw_intensities (
    nucleus_id TEXT NOT NULL,
    time_frame INTEGER NOT NULL CHECK (time_frame >= 0),
    halo1_mcherry REAL,
    halo2_mcherry REAL,
    halo3_mcherry REAL,
    halo4_mcherry REAL,
    halo1_npc REAL,
    halo2_npc REAL,
    halo3_npc REAL,
    halo4_npc REAL,
    halo1_membrane REAL,
    halo2_membrane REAL,
    halo3_membrane REAL,
    halo4_membrane REAL,
    background_mcherry REAL,
    background_npc REAL,
    background_membrane REAL,
    PRIMARY KEY (nucleus_id, time_frame),
    FOREIGN KEY (nucleus_id, time_frame)
        REFERENCES nuclei (nucleus_id, time_frame)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS radial_profile (
    radial_profile_id INTEGER PRIMARY KEY,
    nucleus_id TEXT NOT NULL,
    time_frame INTEGER NOT NULL CHECK (time_frame >= 0),
    theta_deg REAL NOT NULL CHECK (theta_deg >= 0 AND theta_deg < 360),
    rho_normalized REAL NOT NULL CHECK (rho_normalized >= 0),
    w_wall_proximity_index REAL,
    channel TEXT NOT NULL,
    intensity REAL,
    is_ridge_point INTEGER NOT NULL CHECK (is_ridge_point IN (0, 1)),
    cluster_id TEXT,
    UNIQUE (nucleus_id, time_frame, theta_deg, rho_normalized, channel),
    FOREIGN KEY (nucleus_id, time_frame)
        REFERENCES nuclei (nucleus_id, time_frame)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_nuclei_experiment
ON nuclei (experiment_id, fov_id, droplet_id);

CREATE INDEX IF NOT EXISTS ix_nuclei_time
ON nuclei (time_real_minutes);

CREATE INDEX IF NOT EXISTS ix_nuclei_qc
ON nuclei (qc_flag);

CREATE INDEX IF NOT EXISTS ix_radial_profile_lookup
ON radial_profile (nucleus_id, time_frame, channel, theta_deg);

CREATE TRIGGER IF NOT EXISTS validate_selected_slice_insert
BEFORE INSERT ON nucleus_z_stack
WHEN NEW.is_selected_max = 1
BEGIN
    SELECT CASE
        WHEN NEW.slice_id != (
            SELECT selected_slice_id
            FROM nuclei
            WHERE nucleus_id = NEW.nucleus_id
              AND time_frame = NEW.time_frame
        )
        THEN RAISE(ABORT, 'Selected Z-slice does not match nuclei.selected_slice_id')
    END;
END;

CREATE TRIGGER IF NOT EXISTS validate_selected_slice_update
BEFORE UPDATE OF is_selected_max, slice_id ON nucleus_z_stack
WHEN NEW.is_selected_max = 1
BEGIN
    SELECT CASE
        WHEN NEW.slice_id != (
            SELECT selected_slice_id
            FROM nuclei
            WHERE nucleus_id = NEW.nucleus_id
              AND time_frame = NEW.time_frame
        )
        THEN RAISE(ABORT, 'Selected Z-slice does not match nuclei.selected_slice_id')
    END;
END;
