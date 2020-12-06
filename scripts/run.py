import os
import time
import shutil
import urllib.request
import subprocess
import zipfile

project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
data_folder = os.path.join(project_folder, "data")

data_and_results = [
    ["s18.e16.rmat.edgelist.tsv", "kmax = 164, Edges in kmax-truss = 225529."],
    ["s19.e16.rmat.edgelist.tsv", "kmax = 223, Edges in kmax-truss = 334934."],
    ["cit-Patents.tsv", "kmax = 36, Edges in kmax-truss = 2625."],
    ["soc-LiveJournal.tsv", "kmax = 362, Edges in kmax-truss = 72913."],
    ["s18.e16.tsv", "kmax = 164, Edges in kmax-truss = 226780."],
    ["s19.e16.tsv", "kmax = 226, Edges in kmax-truss = 332802."],
    ["complete_graph.tsv", "kmax = 3, Edges in kmax-truss = 3."],
    ["tricount_0.tsv", "kmax = 2, Edges in kmax-truss = 4."],
    ["tricount_1.tsv", "kmax = 3, Edges in kmax-truss = 3."],
]


def download():
    def recu_down(url, filename):
        try:
            urllib.request.urlretrieve(url, filename)
        except urllib.error.ContentTooShortError:
            print('Network conditions is not good. Reloading...')
            recu_down(url, filename)
    os.chdir(project_folder)
    url = "http://datafountain.int-yt.com/Files/BDCI2020/473HuaKeDaKtruss/ktruss-data.zip"
    if not os.path.exists(data_folder):
        start = time.time()
        zip_file_path = "ktruss-data.zip"
        if not os.path.exists(zip_file_path):
            download_start = time.time()
            print("start download")
            recu_down(url, zip_file_path)
            print("download cost: {}s".format(int(time.time() - download_start)))
        with zipfile.ZipFile(zip_file_path) as f:
            f.extractall(path=data_folder)
        for d, _ in data_and_results:
            src_path = os.path.join(data_folder, "ktruss-data", d)
            dst_path = os.path.join(data_folder, d)
            if not os.path.exists(src_path):
                continue
            shutil.move(src_path, dst_path)
        os.rmdir(os.path.join(data_folder, "ktruss-data"))
        print("all cost: {}s".format(int(time.time() - start)))


def kron_gen():
    os.chdir(project_folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(os.path.join(data_folder, "s18.e16.tsv")):
        os.system("./kron_gen 18 16")
    if not os.path.exists(os.path.join(data_folder, "s19.e16.tsv")):
        os.system("./kron_gen 19 16")
    for d, _ in data_and_results:
        src_path = os.path.join(d)
        dst_path = os.path.join(data_folder, d)
        if not os.path.exists(src_path):
            continue
        shutil.move(src_path, dst_path)
    for d, _ in data_and_results:
        src_path = os.path.join("examples", d)
        dst_path = os.path.join(data_folder, d)
        if not os.path.exists(src_path):
            continue
        shutil.move(src_path, dst_path)


def time_analysis(cmd, data_path, result, count=1, para=""):
    t = list()
    for i in range(count):
        args = [os.path.join(project_folder, cmd), "-f", data_path, para]
        start = time.time()
        p = subprocess.Popen(" ".join(args), shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        stdout_output, stderr_output = p.communicate()
        cost_time = (time.time() - start) * 1000
        t.append(cost_time)
        line = stdout_output.decode().strip()
        if line != result:
            print("result is wrong!!!")
            print("{} {} {} {}".format(cmd, data_path, line, result))
            exit(-1)
    # print("cmd: {} data_path: {} para: {} count: {} min: {:.0f}ms max: {:.0f}ms avg: {:.0f}ms median: {:.0f}ms"
    #       .format(cmd, data_path, para, count, min(t), max(t), sum(t) / len(t), sorted(t)[len(t) // 2]))
    return min(t), max(t), sum(t) / len(t), sorted(t)[len(t) // 2]


def main():
    cmds = [
        ["kmax_truss_omp", 5],
        ["kmax_truss_cuda", 5],
        ["kmax_truss_serial", 3]
    ]
    print("project_folder: {}".format(project_folder))
    os.chdir(project_folder)
    # download()
    for cmd in cmds:
        if os.path.exists(cmd[0]):
            os.remove(cmd[0])
    if os.path.exists("kron_gen"):
        os.remove("kron_gen")
    os.system("make -j")
    try:
        os.system("make -j kmax_truss_cuda")
    except Exception as e:
        print(e)
    kron_gen()
    start = time.time()
    for cmd in cmds:
        if not os.path.exists(cmd[0]):
            print("cmd is not exist: {}".format(cmd[0]))
            continue
        for d in data_and_results:
            data_file = os.path.join(data_folder, d[0])
            if not os.path.exists(data_file):
                print("file is not exist: {}".format(data_file))
                continue
            min_t, max_t, _, mid_t = time_analysis(cmd[0], data_file, d[1], cmd[1])
            print("{:^20} {:^30} {:.0f}ms {:.0f}ms {:.0f}ms".format(cmd[0], d[0], min_t, mid_t, max_t))
    print("run cost: {}s".format(int(time.time() - start)))


if __name__ == '__main__':
    main()
