import pandas as pd
import uuid
from collections import OrderedDict

# Columns name of the .csv header according to the standard Eurocontrol template.
columns_name = ['ECTRL ID',
                'Sequence Number',
                'Time Over',
                'Flight Level', # Actually is Z coordinate
                'Latitude',     # Actually is Y coordinate
                'Longitude']    # Actually is X coordinate

def generate_flight_id():
    # Generate a unique ID for a flight.

    id = uuid.uuid1()
    flight_id = id.fields[0]

    return flight_id

okk = generate_flight_id()
print(okk)
def flights_points(data, file_name):
    # 'data' is an ordered list represented in this way --> [ [flights_IDs], [number_waypoints_sequence], [crossingg_waypoints_times], [flights_levels], [Latitudes], [Longitutes] ].
    # This method take 'data' as input and put it into a .csv file (by creating it) according to the Eurocontrol standard template.

    d = {}

    header = True
    for flight in data:
        fields = len(columns_name)
        print(flight)
        for field in range(fields):
            print(d[columns_name[field]])
            d[columns_name[field]] = flight[field]

        df = pd.DataFrame(d)
        df = df[columns_name]
        mode = 'w' if header==True else 'a'
        df.to_csv(file_name, encoding='utf-8', mode=mode, header=header, index=False)
        header = False

ok = [okk, '12', '8', '10', '12', '23']
ok1 = flights_points(ok, 'dam.csv')
def extract_waypoints_from_flights_points_csv(self, file_name):
    # Read a .csv file (according to the Eurocontrol standard template) and return a dictionary in which each key is a flight ID and
    # the corresponding values are the Z,Y,X coordinates of the crossed waypoints for that considered flight.

    n_names = len(columns_name)

    file = pd.read_csv(file_name, header=0)
    flights_and_coords = [[] for i in range(4)] # 4 = flightID + Z + Y + X

    flights_and_coords[0] = file[columns_name[0]].values
    flights_and_coords[1] = file[columns_name[3]].values
    flights_and_coords[2] = file[columns_name[4]].values
    flights_and_coords[3] = file[columns_name[5]].values

    flights_IDs = list(OrderedDict.fromkeys(flights_and_coords[0]))
    n_flights = len(flights_IDs)
    occurrennces_per_flight = [list(flights_and_coords[0]).count(flights_IDs[i]) for i in range(n_flights)]

    previous_occurrence = 0
    current_occurrence = 0
    flights_and_coords_dict = {}
    for i in range(n_flights):
        current_occurrence += occurrennces_per_flight[i]
        flights_and_coords_dict[flights_IDs[i]] = [flights_and_coords[1][previous_occurrence:current_occurrence],
                                                   flights_and_coords[2][previous_occurrence:current_occurrence],
                                                   flights_and_coords[3][previous_occurrence:current_occurrence]]
        previous_occurrence = current_occurrence

    return flights_and_coords_dict