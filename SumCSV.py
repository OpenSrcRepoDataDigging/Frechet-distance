def SumCSV(filename,outputfile):
    with open(filename, 'r') as f:
        with open(outputfile, 'w') as out:
            line = f.readline()
            out.write(line)
            line = line.strip()
            line = line.split(',')
            length = len(line)
            array = [0 for i in range(length)]
            for line in f.readlines():
                out.write(line)
                line = line.strip()
                line = line.split(',')
                for i in range(1, len(line)):
                    array[i] += int(line[i])
            out.write("Summary")
            for i in range(1, len(array)):
                out.write("," + str(array[i]))

SumCSV('files/commitday.csv', 'files/sum_commitday.csv')