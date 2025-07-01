## Install packages (general)
```powershell
$VERSION="3.12";
$ENV_NAME="azfdyagents";
$ENV_SURFIX="pip";

$ENV_FULL_NAME = "$ENV_NAME$VERSION$ENV_SURFIX";
# with the closing "\"
$ENV_DIR="$env:USERPROFILE\Documents\VENV\";

# absolute path of requirements.txt to install for the python venv
$PROJ_DIR="$env:USERPROFILE\Documents\VCS\democollections\AgentEvalExamle";
# $SubProj="\"
$SubProj="fdy\"
$typeProj="_fdy"
$PackageFile="$PROJ_DIR\${SubProj}requirements${typeProj}.txt";

& "$ENV_DIR$ENV_FULL_NAME\Scripts\Activate.ps1";
Invoke-Expression "(Get-Command python).Source";

& "python" -m pip install --upgrade pip;
& "python" -m pip install -r $PackageFile --no-cache-dir;

deactivate
```

## (optional) generate requirements.txt from venv
```powershell
& "python" -m pip freeze > requirements_$(Get-Date -Format "yyyy-MM-dd_HH-mm-ss").txt
```

```powershell
$file1="requirements_2025-06-30_13-12-23.txt"
$file2="requirements_2025-06-30_13-17-09.txt"
$outfile="diffoutput.txt"
Compare-Object (Get-Content $file1) (Get-Content $file2) -IgnoreCase | Out-File $outfile;

```

## (Optional) remove all the packages
For the venv python
```powershell
# which python powershell equivalent
Invoke-Expression "(Get-Command python).Source";
& "python" -m pip freeze | %{$_.split('==')} | %{python -m pip uninstall -y $_};
& "python" -m pip list;
```