@echo off
set DLL=G:\Tri-Core-Orchestrator-ULTRA\CipherCore_OpenCl.dll
python materials_orchestrator_v4.py --epochs 50 --pop 64 --dim 8 --strategy mix --dll "%DLL%" --field-p1 12 --field-p2 0.35 --lr0 0.1 --vqe-steps 8 --seed 42 --log info --vqe auto --num-qubits 4 --ansatz-layers 2 --num-h-terms 2
pause
