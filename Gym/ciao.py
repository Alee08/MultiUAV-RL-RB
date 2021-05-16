import csv
from os import mkdir
from os.path import join, isdir
from datetime import datetime

output = [[[4, 9], [5, 9], [5, 9], [5, 8], [6, 8], [6, 9], [6, 9], [7, 9], [7, 9], [8, 9], [8, 8], [8, 8], [8, 7], [8, 7]]]
traj = []
for i in range(len(output)):
    for j in range(len(output[i])):
        traj.append(output[i][j])
print(traj, "trajjjj")



with open('myfile.csv', 'a', newline='') as file:
    mywriter = csv.writer(file, delimiter='-')
    mywriter.writerows(traj)



with open('myfile.csv', 'r', newline='') as file:
  myreader = csv.reader(file, delimiter='-')
  traj_csv = []
  for rows in myreader:
    traj_csv.append([int(x) for x in rows])
    #print(rows)
print(traj_csv, "dataaaaaaa")

traj = [[8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2]]

for i in range(len(traj)):
    traj[i].insert(0, 20)
print(traj)
out = open('out.csv', 'a')
for row in traj:
    for column in row:
        out.write('%d, ' % column)
    out.write('\n')
out.close()
traj = [[8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2], [8, 2]]

for f in range(len(traj)):
    traj[i].insert(0, 20)
print(traj)


def generate_single_trajectory(csv_filename, csv_dir, data):

    out = open('./' + csv_dir + '/' + csv_filename, 'a')
    print(data)
    for row in data:
        for column in row:
            out.write('%d, ' % column)
        out.write('\n')
    out.close()
CSV_DIRECTORY_NAME = "Flights_trajectories"
if not isdir(CSV_DIRECTORY_NAME): mkdir(CSV_DIRECTORY_NAME)
num_trajectories = 1
def generate_trajectories(name, data):
    csv_dir = str(CSV_DIRECTORY_NAME+"/Fligth_"+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))\
         .replace("-", "_")\
        .replace(":", "_")\
        .replace(".", "")\
        .replace(" ", "--").strip()
        )
    mkdir(csv_dir)
    for i in range(0,num_trajectories):
        csv_filename=csv_dir+"/"+"flight_"+str(i+1)
        generate_single_trajectory(name, csv_dir, data)
    return None



#generate_trajectories('ciao', data)


'''with open('myfile.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)
    #print(list_of_rows)

print(list_of_rows)
a="[[a,b],[c,d]"
without_brackets = a.replace('[','').split('],')
lists = [list(s.replace(']','').split(',')) for s in without_brackets]
print(lists)'''

'''with open("trajectory.csv", "a") as text_file:
    if j == 0:
        start_str = '['
    else:
        start_str = ''

    if j == N_MISSION - 1:
        text_file.write('{}'.format(traj_j) + ']')

    else:
        text_file.write(start_str + '{}'.format(traj_j) + ',')'''

