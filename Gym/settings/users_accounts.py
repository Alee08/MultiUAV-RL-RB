from enum import Enum, IntEnum
#################################################################################################
# Features here below are implemented and insert inside the scenario, but they are not used yed #
#################################################################################################

# -------------- USERS ACCOUNTS --------------:

class UsersAccounts(IntEnum):
    BASE_USER = 1
    FREE_USER = 0
    FULL_USER = 2
    PREMIUM_USER = 3

class UsersAccountsProbs(Enum):
    BASE_USER_PROB = 0.4
    FREE_USER_PROB = 0.3
    FULL_USER_PROB = 0.2
    PREMIUM_USER_PROB = 0.1

USERS_ACCOUNTS = list(map(int, UsersAccounts))
# Probabilities that a user has a specific type of account:
USERS_ACCOUNTS_DITRIBUTIONS = [prob.value for prob in UsersAccountsProbs]

d = {}
for prob in UsersAccountsProbs:
    d[prob.name]= prob.value