from sys import stderr

src_dir = './data/'
target_dir = './fk2/data/'


def gen_fake(src, tgt, fname):
    with open(src + fname) as fin, open(tgt + fname, 'w') as fout:
        line = fin.readline()
        fout.write(line)
        lines = fin.readlines()
        fin.close()
        mn_prc, mx_prc = 1000000.0, 0
        for line in lines:
            f = line.strip().split(',')
            mn_prc = min(mn_prc, float(f[-2]))
            mx_prc = max(mx_prc, float(f[-3]))
        stderr.write('daily min/max price for %s is %f, %f\n' % (fname, mn_prc, mx_prc))
        delta = int(mn_prc * 0.9)

        for line in lines:
            f = line.strip().split(',')
            f[5] = str(float(f[5]) - delta)
            f[-2] = str(float(f[-2]) - delta)
            f[-3] = str(float(f[-3]) - delta)
            fout.write('%s\n' % ','.join(f))


def gen_fake2(src, tgt, fname):
    with open(src + fname) as fin, open(tgt + fname, 'w') as fout:
        line = fin.readline()
        fout.write(line)
        lines = fin.readlines()
        fin.close()
        mn_prc, mx_prc = 1000000.0, 0
        lid = 0
        for line in lines:
            f = line.strip().split(',')
            lno = int(f[0]) + 1
            mn_prc = min(mn_prc, float(f[-2]))
            mx_prc = max(mx_prc, float(f[-3]))
            if (lno % 5) == 0:
                f[-2], f[-3], f[0] = str(mn_prc), str(mx_prc), str(lid)
                fout.write(','.join(f) + '\n')
                mn_prc, mx_prc = 1000000.0, 0
                lid += 1


if __name__ == '__main__':
    for v in ['a', 'i', 'j', 'jm', 'm', 'p', 'y']:
        fname = v + '_minutes_clean.csv'
        gen_fake2(src_dir, target_dir, fname)
