#Making databases for my Nuclei Data. 
require(tidyverse)
require(DBI)
require(RSQLite)

# Create or connect to SQLite database
con <- dbConnect(RSQLite::SQLite(), "nuclear_scaling.db")

# Drop tables if they exist (for testing or rebuilding)
dbExecute(con, "DROP TABLE IF EXISTS measurements")
dbExecute(con, "DROP TABLE IF EXISTS cells")
dbExecute(con, "DROP TABLE IF EXISTS experiments")

# Create experiments table
dbExecute(con, "
  CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    treatment_group TEXT NOT NULL,
    date DATE,
    notes TEXT
  )
")

# Create cells table
dbExecute(con, "
  CREATE TABLE cells (
    cell_id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    cluster INTEGER,
    slice INTEGER,
    time_point INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
  )
")

# Create measurements table
dbExecute(con, "
  CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY,
    cell_id INTEGER NOT NULL,
    N_C_V2 REAL,
    Nuclear_Membrane REAL,
    Nuclear_Membrane_BG_Sub REAL,
    Nuclear_Pores REAL,
    Nuclear_Pores_BG_Sub REAL,
    Nuclear_Area REAL,
    X REAL,
    Y REAL,
    SD REAL,
    FOREIGN KEY (cell_id) REFERENCES cells(cell_id)
  )
")

# Disconnect when done
dbDisconnect(con)

cat("Schema created successfully in 'nuclear_scaling.db'\n")

