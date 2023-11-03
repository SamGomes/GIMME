
$NUM_PROCESSES = 8
for ($i=0; $i -lt $NUM_PROCESSES; $i++)
{
    invoke-expression 'wt -w 0 powershell -Command{
        $myshell = New-Object -ComObject wscript.shell;
        cd C:\Users\samsg\OneDrive\Documents\reps_aux\GIMME;
        clear;
        ..\venv\Scripts\Activate.ps1;
        python .\examples\simulations\simulations.py
    }'
}
