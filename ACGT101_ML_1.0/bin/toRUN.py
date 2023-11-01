
import subprocess
from config import bind_path, sif, parameters_path

# 读取parameter.txt文件，获取cpu和partition参数
with open("/scripts/parameters.txt", "r") as param_file:
    param_lines = param_file.readlines()
    for line in param_lines:
        if "cpu" in line:
            cpu = line.split("=")[1].strip()
        elif "partition" in line:
            partition = line.split("=")[1].strip()

# 生成sh指令
sh_command = f"srun -p {partition} -c {cpu} singularity exec --nv --bind {bind_path} {sif} python3 /scripts/ACGT101_ML_1.0.py -p {parameters_path}"

# 生成.sh文件
sh_file = "/scripts/run_script.sh"
with open(sh_file, "w") as file:
    file.write("#!/bin/bash\n")
    file.write(sh_command)
    
# import subprocess
# import time


# sh=sh_file
# partition=partition

# def run_cmd(job, sh, ls, partition, ed, ld, sd, cpu, node):
    # cpu = 1 if not cpu else cpu
    # if os.path.exists(sh):
        # start = time.time()
        # ret = 0
        # cmd = f"{perl} slurm.pl {sh} {ls} {partition} {ed} {ld} {sd} {cpu} {node}"
        # print(f"Start {sh} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ...", file=sys.stderr)
        
        # ret = subprocess.call(cmd, shell=True)
        
        # end = time.time()
        # dur = end - start
        # status = "Done"
        # if ret != 0:
            # status = "Stop"
        
        # print(f"{status} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.", file=sys.stderr)
        # print(f"Duration: {dur}", file=sys.stderr)
        # if ret != 0:
            # exit(-1)
    
    
# if __name__ == "__main__":
    # run_cmd()


# 赋予.sh文件执行权限

subprocess.run(["perl", "slurm.pl"])

subprocess.run(["chmod", "+x", sh_file])
subprocess.run(["/scripts/run_script.sh"])