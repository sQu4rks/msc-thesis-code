NODES = {
    "SEAT": ("Seattle",        -122.33, 47.61),
    "SNVA": ("Sunnyvale",      -122.04, 37.37),
    "LOSA": ("Los Angeles",    -118.24, 34.05),
    "DENV": ("Denver",         -104.99, 39.74),
    "KSCY": ("Kansas City",     -94.58, 39.10),
    "HOUS": ("Houston",         -95.37, 29.76),
    "CHIN": ("Chicago",         -87.63, 41.88),
    "IPLS": ("Indianapolis",    -86.16, 39.77),
    "ATLA": ("Atlanta",         -84.39, 33.75),
    "WASH": ("Washington DC",   -77.04, 38.91),
    "NYCM": ("New York",        -74.01, 40.71),
}

LINKS = [
    ("SEAT", "SNVA",  7.8,  2000),  
    ("SEAT", "DENV",  9.5,   200), 
    ("SNVA", "LOSA",  3.3,  1500),
    ("SNVA", "DENV",  8.7,   400), 
    ("LOSA", "HOUS", 12.2,   300), 
    ("DENV", "KSCY",  4.8,  1200),
    ("KSCY", "HOUS",  5.5,   500),  
    ("KSCY", "IPLS",  3.8,  1500),  
    ("CHIN", "IPLS",  1.5,  1800),
    ("CHIN", "NYCM",  6.5,   350),  
    ("IPLS", "ATLA",  3.6,  1000),
    ("IPLS", "WASH",  4.2,  1400),
    ("ATLA", "WASH",  4.4,   600), 
    ("WASH", "NYCM",  2.8,  2000),  
]

NODE_WEIGHTS = {
        "SEAT": 0.8,  "SNVA": 1.2,  "LOSA": 1.5,
        "DENV": 0.6,  "KSCY": 0.5,  "HOUS": 0.9,
        "CHIN": 1.3,  "IPLS": 0.7,  "ATLA": 1.0,
        "WASH": 1.1,  "NYCM": 1.4,
}
NODE_LIST = list(NODES.keys())