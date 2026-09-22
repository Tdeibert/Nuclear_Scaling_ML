PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS radial_profile_files (
    radial_file_id INTEGER PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    fov_id TEXT NOT NULL,
    droplet_id TEXT NOT NULL,
    source_csv_path TEXT NOT NULL,
    parquet_path TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    parquet_size_bytes INTEGER NOT NULL CHECK (parquet_size_bytes >= 0),
    compression TEXT NOT NULL DEFAULT 'zstd',
    created_at TEXT NOT NULL,
    UNIQUE (experiment_id, fov_id, droplet_id, parquet_path),
    FOREIGN KEY (experiment_id, fov_id, droplet_id)
        REFERENCES experimental_cfg (experiment_id, fov_id, droplet_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS import_runs (
    import_run_id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    source_directory TEXT NOT NULL,
    experiment_id TEXT,
    fov_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    nuclei_rows INTEGER,
    z_stack_rows INTEGER,
    raw_intensity_rows INTEGER,
    radial_rows INTEGER,
    message TEXT
);

CREATE INDEX IF NOT EXISTS ix_radial_files_experiment
ON radial_profile_files (experiment_id, fov_id);

CREATE INDEX IF NOT EXISTS ix_import_runs_experiment
ON import_runs (experiment_id, fov_id, started_at);
