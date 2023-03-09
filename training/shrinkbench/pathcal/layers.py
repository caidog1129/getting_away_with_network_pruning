import re

def isConv(layer):
    c = str(type(layer)).lower()
    return re.search(r'.conv*', c) != None

def isLinear(layer):
    c = str(type(layer)).lower()
    return re.search(r'.linear*', c) != None