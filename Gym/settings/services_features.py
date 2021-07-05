from enum import Enum, IntEnum

# -------------------------------------------------------------- USERS TYPE OF REQUESTS --------------------------------------------------------------

# USERS THROUGHPUT REQUESTS [Kb/s]:
class ThroughputRequests(IntEnum, Enum):
    AUDIO_STREAM_NORMAL = 96
    AUDIO_STREAM_HIGH = 160
    AUDIO_STREAM_EXTREME = 320
    VIDEOCALL = 200
    YOUTUBE_360p = 750
    YOUTUBE_480p = 1000
    YOUTUBE_720p = 2500    

# USERS THROUGHPUT REQUESTS TIMES [iterations]:
class ThroughputRequestsTimes(IntEnum, Enum):
    TR_SERVICE_TIME1 = 1
    TR_SERVICE_TIME2 = 2
    TR_SERVICE_TIME3 = 3
    TR_SERVICE_TIME4 = 4
    TR_SERVICE_TIME5 = 5
    TR_SERVICE_TIME6 = 6

# USERS EDGE-COMPUTING REQUESTS [Bytes]:
class EdgeComputingRequests(IntEnum, Enum):
    BYTE1 = 10
    BYTE2 = 20
    BYTE3 = 30
    BYTE4 = 40
    BYTE5 = 50

# USERS MESSAGES FOR DATA GATHERING [# Messages]:
class DataGatheringRequests(IntEnum, Enum):      
    MESSAGE1 = 3
    MESSAGE2 = 6
    MESSAGE3 = 9
    MESSAGE4 = 12
    MESSAGE5 = 24

# POSSIBLE SERVICES:
class UsersServices(IntEnum, Enum):
    NO_SERVICE = 0
    THROUGHPUT_REQUEST = 1
    EDGE_COMPUTING = 2
    DATA_GATHERING = 3

# SERVICES PROBABILITIES TO BE ASKED FROM USERS:
class UsersServicesProbs(Enum):
    NO_SERVICE_PROB = 0.1
    THROUGHPUT_REQUEST_PROB = 0.5
    EDGE_COMPUTING_PROB = 0.25
    DATA_GATHERING_PROB = 0.15

def byte_to_bit(byte):
    return byte/8

def bit_to_kb(bit):
    return bit*1024

def message_to_bit(n_message):
    # Each (data-gathering) message is worth 8 bits: 
    return n_message*8

TRHOUGHPUT_REQUESTS = list(map(int, ThroughputRequests))
TR_SERVICE_TIMES = list(map(int, ThroughputRequestsTimes))

EDGE_COMPUTING_REQUESTS = list(map(int, EdgeComputingRequests))
EDGE_COMPUTING_REQUESTS = [bit_to_kb(byte_to_bit(req)) for req in EDGE_COMPUTING_REQUESTS]
# USERS FOR EDGE-COMPUTING TIME [iterations]:
EC_SERVICE_TIME = 3

DATA_GATHERING_REQUESTS = list(map(int, DataGatheringRequests))
DATA_GATHERING_REQUESTS = [bit_to_kb(message_to_bit(message)) for message in DATA_GATHERING_REQUESTS]
# USERS DATA-GATHERING TIME [iterations]:
DG_SERVICE_TIME = 2

# Expected services from usere to UAVs:
UAVS_SERVICES = list(map(int, UsersServices))
# Probabilities related to the each service to be asked from users:
SERVICE_PROBABILITIES = [prob.value for prob in UsersServicesProbs]