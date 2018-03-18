sp="/raid/tools/SP/SystemProfiler-linux-public-4.0.1163-b6fa40c/Target-x86_64/x86_64/sp"
cmd="python3 main.py"
cd ..
rm -fr ./*.qdstrm
$sp profile --delay=10 --duration=5 -o pytorch-capsule.qdstrm -t cuda,cublas,curand,system,cudnn $cmd
