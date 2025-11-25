# Windows Setup Guide - Rock Glacier Deformation Analysis

## Overview

This guide provides **Windows-specific** installation instructions for the rock glacier deformation workflow. Since COSI-Corr's installation script uses Linux/bash commands, we'll do the setup manually.

---

## System Requirements

- **OS**: Windows 10 or 11
- **Storage**: ~5 GB for environment + tools
- **RAM**: 8+ GB recommended (16+ GB for full DEMs)
- **Python**: Will be installed via Mamba/Conda

---

## Step-by-Step Installation

### Step 1: Open Anaconda Prompt

1. Press `Windows Key`
2. Type "Anaconda Prompt"
3. Right-click → **Run as Administrator** (recommended)

You should see a command prompt that looks like:
```
(base) C:\Users\YourName>
```

---

### Step 2: Activate Your Mamba Environment

Since your Mamba is installed in a custom environment (not base), activate it first:

```cmd
conda activate mamba-env
```

**Verify it worked:**
- Your prompt should change to: `(mamba-env) C:\Users\YourName>`
- Test mamba: `mamba --version`
  - Should output something like: `mamba 1.5.x`

**If this fails:**
```cmd
# Alternative: Activate using full path
conda activate C:\Users\mmorriss\Anaconda3\envs\mamba-env
```

---

### Step 3: Create the Rock Glacier Analysis Environment

Now use Mamba to create a new environment with all required packages:

```cmd
mamba create -n rock_glacier_env python=3.9 ^
    gdal rasterio geopandas numpy matplotlib scipy ^
    xdem joblib scikit-image pandas jupyterlab ^
    -c conda-forge
```

**What this does:**
- Creates new environment named `rock_glacier_env`
- Installs Python 3.9 (required for COSI-Corr compatibility)
- Installs all geospatial libraries from conda-forge channel

**Note:** The `^` symbol is the Windows line continuation character. You can also type it all on one line.

**When prompted** `Proceed ([y]/n)?` → Type `y` and press Enter

**Installation time:** ~5-10 minutes (downloads ~1-2 GB)

---

### Step 4: Activate the New Environment

```cmd
mamba activate rock_glacier_env
```

Your prompt should now show:
```
(rock_glacier_env) C:\Users\YourName>
```

---

### Step 5: Verify Python Installation

```cmd
python --version
```
Should output: `Python 3.9.x`

```cmd
python -c "import xdem; import rasterio; import geopandas; print('✓ All packages loaded')"
```
Should output: `✓ All packages loaded`

**If you get errors**, see [Troubleshooting](#troubleshooting) below.

---

### Step 6: Install COSI-Corr (Manual Method for Windows)

Since the COSI-Corr install script uses bash commands, we'll download it manually.

#### Option A: Manual Download (Recommended for Windows)

1. **Open your web browser**
2. Go to: https://github.com/SaifAati/Geospatial-COSICorr3D
3. Click the green **"Code"** button → **"Download ZIP"**
4. Save to your Downloads folder
5. **Extract the ZIP file**:
   - Right-click `Geospatial-COSICorr3D-main.zip`
   - Choose "Extract All..."
   - Extract to: `M:\My Drive\Rock Glaciers\Tools\`
6. **Rename the folder**:
   - The extracted folder will be named `Geospatial-COSICorr3D-main`
   - Rename it to `Geospatial-COSICorr3D` (remove the `-main` suffix)

#### Option B: Using Git for Windows

If you have Git installed:

```cmd
cd "M:\My Drive\Rock Glaciers\Tools"
git clone https://github.com/SaifAati/Geospatial-COSICorr3D.git
```

---

### Step 7: Verify COSI-Corr Installation

Check that the correlation script exists:

```cmd
dir "M:\My Drive\Rock Glaciers\Tools\Geospatial-COSICorr3D\geoCosiCorr3D\scripts\correlation.py"
```

**Expected output:** File details (size, date, etc.)

**If "File Not Found":**
- Double-check the extraction location
- Verify the folder wasn't renamed incorrectly
- Make sure you extracted the **inner** folder (Geospatial-COSICorr3D)

---

### Step 8: Update Script Paths

The scripts are pre-configured with example paths. You need to update them to match your system.

#### In `01_horizontal_displacement_cosicorr.py`:

Find this line (~Line 37):
```python
COSICORR_SCRIPT = r"M:/My Drive/Rock Glaciers/Tools/Geospatial-COSICorr3D/geoCosiCorr3D/scripts/correlation.py"
```

**Verify the path exists** by opening File Explorer and navigating to:
```
M:\My Drive\Rock Glaciers\Tools\Geospatial-COSICorr3D\geoCosiCorr3D\scripts\correlation.py
```

**Important Notes about Windows Paths in Python:**
- Use `r"..."` (raw string) to avoid backslash issues
- You can use **forward slashes** `/` even on Windows (Python handles it)
- Or use **double backslashes** `\\` if you prefer:
  ```python
  COSICORR_SCRIPT = r"M:\My Drive\Rock Glaciers\Tools\Geospatial-COSICorr3D\geoCosiCorr3D\scripts\correlation.py"
  # OR
  COSICORR_SCRIPT = "M:/My Drive/Rock Glaciers/Tools/Geospatial-COSICorr3D/geoCosiCorr3D/scripts/correlation.py"
  ```

---

## Testing the Installation

### Test 1: Python Environment

```cmd
mamba activate rock_glacier_env

python -c "import numpy; import rasterio; import geopandas; import xdem; print('Environment OK')"
```

Expected: `Environment OK`

### Test 2: COSI-Corr Accessibility

```cmd
python -c "from pathlib import Path; p = Path(r'M:/My Drive/Rock Glaciers/Tools/Geospatial-COSICorr3D/geoCosiCorr3D/scripts/correlation.py'); print('COSI-Corr found!' if p.exists() else 'NOT FOUND')"
```

Expected: `COSI-Corr found!`

### Test 3: Run a Simple Script

Navigate to your code directory:
```cmd
cd "M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\rg_velocity"
```

Run the DEM characterization (doesn't require COSI-Corr):
```cmd
python 03_dem_quality_assessment.py
```

**Expected behavior:**
- Should start processing
- If DEMs don't exist, you'll get a "File not found" error (that's OK for now)
- If it runs, you'll see progress messages

---

## Environment Management

### Activating the Environment (Every Time You Work)

```cmd
# Option 1: If mamba-env is already active
mamba activate rock_glacier_env

# Option 2: From base environment
conda activate mamba-env
mamba activate rock_glacier_env
```

### Deactivating the Environment

```cmd
conda deactivate
```

### Listing Your Environments

```cmd
conda env list
```

Should show:
```
# conda environments:
#
base                  *  C:\Users\mmorriss\Anaconda3
mamba-env                C:\Users\mmorriss\Anaconda3\envs\mamba-env
rock_glacier_env         C:\Users\mmorriss\Anaconda3\envs\rock_glacier_env
```

### Removing the Environment (If You Need to Start Over)

```cmd
mamba deactivate
mamba env remove -n rock_glacier_env
```

Then repeat Step 3 to recreate it.

---

## Troubleshooting

### Issue 1: "mamba: command not found"

**Cause:** Mamba environment not activated

**Solution:**
```cmd
conda activate mamba-env
```

Verify:
```cmd
mamba --version
```

---

### Issue 2: "'conda' is not recognized as an internal or external command"

**Cause:** Anaconda not in system PATH

**Solution:**

1. **Find your Anaconda installation:**
   - Typical location: `C:\Users\YourName\Anaconda3\`
   - Or: `C:\ProgramData\Anaconda3\`

2. **Open Anaconda Prompt from Start Menu** instead of regular CMD
   - Search "Anaconda Prompt" in Start Menu

3. **Or add to PATH manually** (Advanced):
   - Right-click "This PC" → Properties → Advanced System Settings
   - Environment Variables → Path → Edit
   - Add: `C:\Users\YourName\Anaconda3\Scripts`

---

### Issue 3: "ImportError: DLL load failed" when importing packages

**Cause:** Missing Visual C++ Redistributables

**Solution:**

Download and install:
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

After installation, restart Anaconda Prompt and try again.

---

### Issue 4: "Permission Denied" when creating environment

**Cause:** Insufficient permissions

**Solution:**

1. Close Anaconda Prompt
2. Right-click "Anaconda Prompt" → **Run as Administrator**
3. Repeat Step 3 (create environment)

---

### Issue 5: COSI-Corr script not found

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'M:/My Drive/.../correlation.py'
```

**Diagnosis:**

1. Open File Explorer
2. Navigate to: `M:\My Drive\Rock Glaciers\Tools\`
3. Check if `Geospatial-COSICorr3D` folder exists
4. Inside, check for `geoCosiCorr3D\scripts\correlation.py`

**Common mistakes:**
- Folder named `Geospatial-COSICorr3D-main` instead of `Geospatial-COSICorr3D`
- Extracted to wrong location
- Double-nested folder (extracted ZIP into another folder)

**Fix:**
- Ensure path exactly matches: `M:\My Drive\Rock Glaciers\Tools\Geospatial-COSICorr3D\geoCosiCorr3D\scripts\correlation.py`
- Update `COSICORR_SCRIPT` variable in `01_horizontal_displacement_cosicorr.py` to match actual location

---

### Issue 6: "UnicodeDecodeError" or path issues with spaces

**Cause:** Windows path with spaces (e.g., "My Drive")

**Solution:**

Always use one of these formats:
```python
# Option 1: Raw string (recommended)
path = r"M:\My Drive\Rock Glaciers\..."

# Option 2: Forward slashes (also works on Windows)
path = "M:/My Drive/Rock Glaciers/..."

# Option 3: Double backslashes
path = "M:\\My Drive\\Rock Glaciers\\..."
```

**In scripts, verify all path definitions use `r"..."` prefix.**

---

### Issue 7: "OSError: [WinError 206] The filename or extension is too long"

**Cause:** Windows path length limit (260 characters)

**Solutions:**

1. **Use shorter paths** (recommended):
   ```
   Instead of: M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\
   Use: M:\RG\Snowbird\preprocessed_dems\
   ```

2. **Enable long paths** (Windows 10+):
   - Run as Administrator: `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f`
   - Restart computer

---

## Quick Reference: Common Commands

### Environment Activation (Start of Every Session)
```cmd
conda activate mamba-env
mamba activate rock_glacier_env
cd "M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\rg_velocity"
```

### Run Analysis
```cmd
# Generate test patches (one-time)
python 00_generate_test_patches.py

# Run correlation
python 01_horizontal_displacement_cosicorr.py

# Run 3D synthesis
python 02_3d_synthesis_backwarp.py
```

### Check Environment
```cmd
# List packages
mamba list

# Check Python version
python --version

# Find environment location
conda env list
```

---

## Next Steps

Once installation is complete:

1. **Read WORKFLOW_GUIDE.md** for detailed methodology
2. **Update paths** in all scripts to match your system:
   - DEM file locations
   - Shapefile locations
   - COSI-Corr script path
3. **Generate test patches** (faster iteration):
   ```cmd
   python 00_generate_test_patches.py
   ```
4. **Run on test patch first** (set `USE_TEST_PATCH = True` in scripts)
5. **Review outputs** before running on full dataset

---

## Additional Resources

### Documentation
- Main workflow: `WORKFLOW_GUIDE.md` (comprehensive theory and methods)
- Script comments: Each `.py` file has detailed inline documentation

### Getting Help
- **COSI-Corr issues**: https://github.com/SaifAati/Geospatial-COSICorr3D/issues
- **xdem documentation**: https://xdem.readthedocs.io
- **Conda troubleshooting**: https://docs.conda.io/projects/conda/en/latest/user-guide/troubleshooting.html

### Useful Links
- Anaconda documentation: https://docs.anaconda.com/
- Mamba documentation: https://mamba.readthedocs.io/
- GDAL Windows binaries: https://www.gisinternals.com/release.php

---

## Common Workflow on Windows

```cmd
:: 1. Start Anaconda Prompt

:: 2. Activate environments
conda activate mamba-env
mamba activate rock_glacier_env

:: 3. Navigate to code
cd "M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\rg_velocity"

:: 4. Run analysis
python 01_horizontal_displacement_cosicorr.py

:: 5. When done
conda deactivate
```

---

**Last Updated:** 2025-11-22
**Tested On:** Windows 10/11 with Anaconda3
