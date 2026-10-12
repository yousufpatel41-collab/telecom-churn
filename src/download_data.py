"""Download the IBM sample and verify the pinned CSV checksum."""
import hashlib
from urllib.request import urlopen
from preprocess import DATA_PATH

EXPECTED_SHA="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
if __name__=='__main__':
    raw=DATA_PATH.read_bytes() if DATA_PATH.exists() else urlopen(URL,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA:
        raise SystemExit('Dataset differs from the pinned sample. Review the source before replacing it.')
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(raw)
    print('Verified IBM sample CSV.')
