# Nuclear Scaling SQLite setup

These scripts create and populate a SQLite database for the five data sheets in
`Nuclear_Scaling_Database_Schema.xlsx`.

The default database path is:

```text
/home/tdeibert/Projects/Nuclear_Scaling/nuclear_scaling.db
```

This is written in Python as `~/Projects/Nuclear_Scaling/nuclear_scaling.db`, so
it remains valid when your home directory is mounted differently.

## Install the Excel reader

The database builder itself uses only Python's standard library. Excel import
requires `openpyxl`:

```bash
python -m pip install openpyxl
```

If you use Conda, you may instead run:

```bash
conda install openpyxl
```

## Copy the tools into the project

Suggested location:

```text
~/Projects/Nuclear_Scaling/database_tools/
```

Keep `create_database.py`, `import_workbook.py`, and `schema.sql` together.

## Create an empty database

```bash
cd ~/Projects/Nuclear_Scaling/database_tools
python create_database.py
```

The command is safe to rerun. It creates missing tables and indexes without
deleting existing data.

## Import the workbook

```bash
python import_workbook.py /path/to/Nuclear_Scaling_Database_Schema.xlsx
```

Existing rows with the same declared key are updated; new rows are inserted.
The complete import runs as one transaction. If any row fails validation, none
of that import is committed.

To use a different database path:

```bash
python import_workbook.py /path/to/workbook.xlsx \
  --database /path/to/another/nuclear_scaling.db
```

## Open it in DBeaver

Create a new SQLite connection and select:

```text
/home/tdeibert/Projects/Nuclear_Scaling/nuclear_scaling.db
```

The importer enables foreign keys for its connection. In any separate SQLite
connection, including DBeaver, run:

```sql
PRAGMA foreign_keys = ON;
```

Verify with:

```sql
PRAGMA foreign_keys;
PRAGMA integrity_check;
PRAGMA foreign_key_check;
```

The first query should return `1`, the second `ok`, and the third no rows.

## Identity assumption

`Nucleus_ID` must be globally unique across experiments, FOVs, and droplets.
For example, prefer an identifier such as:

```text
EXP2026-06-12_FOV02_D003_N0001
```

If IDs restart at `N0001` in each experiment, update the pipeline before
importing production data; otherwise child-table relationships can become
ambiguous.
