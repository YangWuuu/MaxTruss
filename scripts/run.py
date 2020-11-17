import subprocess
import os

project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
data_folder = os.path.join(project_folder, "data")


def prepare():
    # download file
    pass


def time_analysis(cmd, data_path, result, count=5, para=""):
    t = list()
    for i in range(count):
        # time.sleep(0.5)
        # print(i, "perf stat {}/{} -f {} {}".format(project_folder, cmd, data_path, para))
        p = subprocess.Popen("perf stat {}/{} -f {} {}".format(project_folder, cmd, data_path, para), shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, 'b'):
            # for line in p.stdout.readlines():
            line = line.decode().strip()
            # print(line)
            if "Edges in kmax-truss = " in line:
                if line != result:
                    print("result is wrong!!!")
                    print("{} {} {} {}".format(cmd, data_path, line, result))
                    exit(-1)
            if "seconds time elapsed" in line:
                t.append(float(line.split(' ')[0]) * 1000)
                break
    print("cmd: {} data_path: {} para: {} count: {} min: {:.0f}ms max: {:.0f}ms avg: {:.0f}ms median: {:.0f}ms"
          .format(cmd, data_path, para, count, min(t), max(t), sum(t) / len(t), sorted(t)[len(t) // 2]))
    return min(t), max(t), sum(t) / len(t)


def main():
    cmds = ["kmax_truss", "kmax_truss_serial"]
    print("project_folder: {}".format(project_folder))
    os.chdir(project_folder)
    for cmd in cmds:
        if os.path.exists(cmd):
            os.remove(cmd)
    os.system("make")
    data = [
        ["s18.e16.rmat.edgelist.tsv", "kmax = 164, Edges in kmax-truss = 225529."],
        ["s19.e16.rmat.edgelist.tsv", "kmax = 223, Edges in kmax-truss = 334934."],
        ["cit-Patents.tsv", "kmax = 36, Edges in kmax-truss = 2625."],
        ["soc-LiveJournal.tsv", "kmax = 362, Edges in kmax-truss = 72913."],
    ]
    for cmd in cmds:
        for d in data:
            data_file = os.path.join(data_folder, d[0])
            if not os.path.exists(data_file):
                print("file is not exist: {}".format(data_file))
            time_analysis(cmd, data_file, d[1])


if __name__ == '__main__':
    main()
