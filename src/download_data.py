"""Download IBM's public sample data; do not overwrite an existing dataset."""
from urllib.request import urlopen
from preprocess import DATA_PATH

if __name__ == "__main__":
    DATA_PATH.parent.mkdir(exist_ok=True)
    if DATA_PATH.exists():
        raise SystemExit("Dataset already exists: " + str(DATA_PATH))
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    with urlopen(url, timeout=60) as response:
        DATA_PATH.write_bytes(response.read())
    print("Downloaded " + str(DATA_PATH))
