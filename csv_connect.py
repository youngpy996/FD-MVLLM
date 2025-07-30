import csv
import glob

csv_files = glob.glob(r'E:\young\JNU-Bearing-Dataset-main/normal/*.csv')
output_file = 'merged normal.csv'

with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    # 处理第一个文件（保留表头）
    with open(csv_files[0], 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            writer.writerow(row)

    # 处理其他文件（跳过表头）
    for file in csv_files[1:]:
        with open(file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)  # 跳过表头
            for row in reader:
                writer.writerow(row)
