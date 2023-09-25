import subprocess, os
import argparse
import socket

def main(args):
    host_name = socket.gethostname()
    logs_path_src = f"/home/{args.user}/a1-loco/legged_gym/logs/parkour_new"
    logs_path_dst = "/home/cxx/a1-loco/legged_gym/logs/parkour_new"
    
    folders = subprocess.check_output(["ssh", f"cxx@{args.server}.pc.cs.cmu.edu", "cd", logs_path_src, "&&", "find", "*", "-maxdepth", "0", "-type", "d"]).decode("utf-8")
    folder_list = folders.split("\n")
    for name in folder_list:
        if len(name) >= 6:
            if name[:6] == args.exptid:
                exp_path_src = os.path.join(logs_path_src, name)
                break
    models = subprocess.check_output(["ssh", f"cxx@{args.server}.pc.cs.cmu.edu", "cd", exp_path_src, "&&", "find", "*", "-maxdepth", "0"]).decode("utf-8")
    models = models.split("\n")
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    model = models[-1]
    if args.ckpt:
        model = f"model_{args.ckpt}.pt"
    model_path_src = os.path.join(exp_path_src, model)
    model_path_dst = os.path.join(logs_path_dst, name, model)
    os.makedirs(os.path.dirname(model_path_dst), exist_ok=True)

    print(f"Copying from: {args.server:}{model_path_src}\nCopying to: {host_name:}{model_path_dst}")
    p = subprocess.Popen(["rsync", "-avz",
                         f"cxx@{args.server}.pc.cs.cmu.edu:" + model_path_src, 
                         model_path_dst])
    sts = os.waitpid(p.pid, 0)

parser = argparse.ArgumentParser()
parser.add_argument('--exptid', type=str, required=True, default='000-00')
parser.add_argument('--server', type=str, required=True, default='vision0')
parser.add_argument('--user', type=str, required=False, default='cxx')
parser.add_argument('--ckpt', type=str, required=False, default='')



args = parser.parse_args()
main(args)

# sts = os.waitpid(p.pid, 0)