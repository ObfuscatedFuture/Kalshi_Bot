
from datetime import datetime

# Transcript paths
snow_paths = [
    'dataset/SNOW-earnings/1Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q25-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q25-transcript-SNOW.pdf'
]
hd_paths = [
    'dataset/HD-earnings/1Q23-transcript-HD.pdf',
    'dataset/HD-earnings/2Q23-transcript-HD.pdf',
    'dataset/HD-earnings/3Q23-transcript-HD.pdf',
    'dataset/HD-earnings/4Q23-transcript-HD.pdf',
    'dataset/HD-earnings/1Q24-transcript-HD.pdf',
    'dataset/HD-earnings/2Q24-transcript-HD.pdf',
    'dataset/HD-earnings/3Q24-transcript-HD.pdf',
    'dataset/HD-earnings/4Q24-transcript-HD.pdf',
    'dataset/HD-earnings/1Q25-transcript-HD.pdf',
    'dataset/HD-earnings/2Q25-transcript-HD.pdf',
    'dataset/HD-earnings/3Q25-transcript-HD.pdf',
]

lowes_paths = [
    'dataset/Lowe-earnings/1Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/3Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/4Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/1Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/3Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/4Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/1Q25-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q25-transcript-LOWE.pdf'
]

target_paths = [
    'dataset/Target-earnings/1Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/3Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/4Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/1Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/3Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/4Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/1Q25-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q25-transcript-TGT.pdf'
]

walmart_paths = [
    'dataset/Walmart-earnings/1Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/3Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/4Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/1Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/3Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/4Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/1Q25-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q25-transcript-WMT.pdf'
]
salesforce_paths = [
    'dataset/CRM-earnings/1Q22-transcript-CRM.pdf',
    'dataset/CRM-earnings/2Q22-transcript-CRM.pdf',
    'dataset/CRM-earnings/3Q22-transcript-CRM.pdf',
    'dataset/CRM-earnings/4Q22-transcript-CRM.pdf',
    'dataset/CRM-earnings/1Q23-transcript-CRM.pdf',
    'dataset/CRM-earnings/2Q23-transcript-CRM.pdf',
    'dataset/CRM-earnings/3Q23-transcript-CRM.pdf',
    'dataset/CRM-earnings/4Q23-transcript-CRM.pdf',
    'dataset/CRM-earnings/1Q24-transcript-CRM.pdf',
    'dataset/CRM-earnings/2Q24-transcript-CRM.pdf',
    "dataset/CRM-earnings/3Q24-transcript-CRM.pdf",
    "dataset/CRM-earnings/4Q24-transcript-CRM.pdf",
    "dataset/CRM-earnings/1Q25-transcript-CRM.pdf",
    "dataset/CRM-earnings/2Q25-transcript-CRM.pdf"
]   

REPORT_DATES = {
    "snowflake": { # done
        "1Q21": "2021-05-26",
        "2Q21": "2021-08-25",
        "3Q21": "2021-12-01",#
        "4Q21": "2022-03-02",#
        "1Q22": "2022-05-25",
        "2Q22": "2022-08-24",
        "3Q22": "2022-11-30", #
        "4Q22": "2023-03-01", #
        "1Q23": "2023-05-24",
        "2Q23": "2023-08-23",
        "3Q23": "2023-11-29", #
        "4Q23": "2024-02-28", #
        "1Q24": "2024-05-22",
        "2Q24": "2024-08-21",
        "3Q24": "2024-11-20",
        "4Q24": "2025-02-26", #
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-27"
    },
    "walmart": { # Done
        "1Q23": "2023-05-18",
        "2Q23": "2023-08-17",
        "3Q23": "2023-11-16",
        "4Q23": "2024-02-20",
        "1Q24": "2024-05-16",
        "2Q24": "2024-08-15",
        "3Q24": "2024-11-19",
        "4Q24": "2025-02-20",
        "1Q25": "2025-05-15",
        "2Q25": "2025-08-21"
    },
    "target": { # Done
        "1Q23": "2023-05-17",
        "2Q23": "2023-08-16",
        "3Q23": "2023-11-15",
        "4Q23": "2024-03-05",
        "1Q24": "2024-05-22",
        "2Q24": "2024-08-22",
        "3Q24": "2024-11-20",
        "4Q24": "2024-03-04",
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-20"
    },
    "home_depot": { # Done
        "1Q23": "2023-05-16",
        "2Q23": "2023-08-15",
        "3Q23": "2023-11-14",
        "4Q23": "2024-02-20",
        "1Q24": "2024-05-14",
        "2Q24": "2024-08-13",
        "3Q24": "2024-11-12",
        "4Q24": "2025-02-25",
        "1Q25": "2025-05-20",
        "2Q25": "2025-08-19",
        "3Q25": "2025-11-18"
    },
    "lowes": { # Done
        "1Q23": "2023-05-23",
        "2Q23": "2023-08-22",
        "3Q23": "2023-11-21",
        "4Q23": "2024-02-27",
        "1Q24": "2024-05-21",
        "2Q24": "2024-08-20",
        "3Q24": "2024-11-19",
        "4Q24": "2025-02-26",
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-20",
    },
    "salesforce": { # TODO: ADD REAL DATES
        "1Q22": "2022-05-24",
        "2Q22": "2022-08-23",
        "3Q22": "2022-11-29",
        "4Q22": "2023-02-28",
        "1Q23": "2023-05-23",
        "2Q23": "2023-08-22",
        "3Q23": "2023-11-28",
        "4Q23": "2024-02-27",
        "1Q24": "2024-05-21",
        "2Q24": "2024-08-20",
        "3Q24": "2024-11-19",
        "4Q24": "2025-02-26",
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-27"
    }
}
# Walmart EPS surprise data (as decimals)
WALMART_EPS_SURPRISE = {
    "1Q23": 0.1136,
    "2Q23": 0.0888,
    "3Q23": 0.0000,
    "4Q23": 0.0909,
    "1Q24": 0.1538,
    "2Q24": 0.0308,
    "3Q24": 0.0943,
    "4Q24": 0.0154,
    "1Q25": 0.0702,
    "2Q25": -0.0685
}

HOME_DEPOT_EPS_SURPRISE = {
    "1Q23": -0.052,
    "2Q23": -0.0409,
    "3Q23": -0.0131,
    "4Q23": -0.0177,
    "1Q24": -0.0055,
    "2Q24": -0.0278,
    "3Q24": -0.0370,
    "4Q24": -0.0288,
    "1Q25": 0.0084,
    "2Q25": 0.0064,
    "3Q25": -0.0184
}

TARGET_EPS_SURPRISE = {
    "1Q23": 0.1752,
    "2Q23": 0.2766,
    "3Q23": 0.4189,
    "4Q23": 0.2365,
    "1Q24": 0.0098,
    "2Q24": 0.1898,
    "3Q24": -0.1921,
    "4Q24": 0.0711,
    "1Q25": -0.1975,
    "2Q25": -0.0191,
    "3Q25": 0.0114
}

LOWES_EPS_SURPRISE = {
    "1Q23": 0.0546,
    "2Q23": 0.0156,
    "3Q23": 0.0033,
    "4Q23": 0.0536,
    "1Q24": 0.0408,
    "2Q24": 0.0354,
    "3Q24": 0.0248,
    "4Q24": 0.0546,
    "1Q25": 0.0139,
    "2Q25": 0.0236,
    "3Q25": 0.0303
}

SNOW_EPS_SURPRISE = {
    "1Q21": -0.5, # TODO: THESE ARE RANDOM AI NUMBERs
    "2Q21": 0.73,
    "3Q21": 1.66,
    "4Q21": 3.00,
    "1Q22": 2.00,
    "2Q22": 1.5,
    "3Q22": 2.667,
    "4Q22": 3.667,
    "1Q23": 2,
    "2Q23": 3.667,
    "3Q23": 0.667,
    "4Q23": 1.058,
    "1Q24": -0.1765,
    "2Q24": 0.2,
    "3Q24": 0.33,
    "4Q24": 0.765,
    "1Q25": 0.0909,
    "2Q25": 0.3462
}

CRM_EPS_SURPRISE = { #TODO Add real data
    "1Q22": 0.02,
    "2Q22": 0.03,
    "3Q22": 0.025,
    "4Q22": 0.04,
    "1Q23": 0.035,
    "2Q23": 0.045,
    "3Q23": 0.05,
    "4Q23": 0.06,
    "1Q24": 0.055,
    "2Q24": 0.065,
    "3Q24": 0.07,
    "4Q24": 0.075,
    "1Q25": 0.08,
    "2Q25": 0.085
}

# Companies config: add more companies by extending this dict
COMPANIES = {
    "walmart": {
        "paths": walmart_paths,
        "eps_surprise": WALMART_EPS_SURPRISE
    },
    "lowes": {
        "paths": lowes_paths,
        "eps_surprise": None
    },
    "home_depot": {
        "paths": hd_paths,
        "eps_surprise": HOME_DEPOT_EPS_SURPRISE
    },
    "target": {
        "paths": target_paths,
        "eps_surprise": TARGET_EPS_SURPRISE
    },
    "snowflake": {
        "paths": snow_paths,
        "eps_surprise": SNOW_EPS_SURPRISE
    },
    "salesforce": {
        "paths": salesforce_paths,
        "eps_surprise": CRM_EPS_SURPRISE
    }
}

TARGET = "snowflake"
COMPETITORS = []

def get_report_date(company, quarter):
    return datetime.fromisoformat(REPORT_DATES[company][quarter])