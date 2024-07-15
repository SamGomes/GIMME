
# 
# konsole --noclose --new-tab -e echo Hello terminal 1! &
# konsole --noclose --new-tab -e echo Hello terminal 2! &
# konsole --noclose --new-tab -e echo Hello terminal 3! &
# konsole --noclose --new-tab -e echo Hello terminal 4! &

numProcesses=5
for i in $(seq 1 $numProcesses) 
do
    konsole --noclose --new-tab -e python3 simulations.py &
done
