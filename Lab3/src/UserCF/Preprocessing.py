from pathlib import Path

with open(Path(__file__).parent.parent.parent.joinpath('DoubanMusic.txt'),'r') as f:
    content = f.readlines()
    record = dict()
    for line in content:
        data = line.strip().split("\t")
        for index in range(1, len(data)):
            val = int(data[index].split(',')[0])
            record[val] = 1
    i = 0
    while i in record:
        i += 1
    print(i)