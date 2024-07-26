import signal

from contextlib import contextmanager

import requests

# Original Lines from Udacity Project
# DELAY = INTERVAL = 4 * 60  # interval time in seconds
# MIN_DELAY = MIN_INTERVAL = 2 * 60
# KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
# TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
# TOKEN_HEADERS = {"Metadata-Flavor":"Google"}

############################## Added alternative lines to run directly on google Colab #################

import time

# Adjusted values for Google Colab
DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60

# Remove or adjust KEEPALIVE_URL if not needed in Colab
KEEPALIVE_URL = None  # or comment out this line if not applicable

# If you need to use a token for some API, you should provide your own URL and headers
TOKEN_URL = None  # or your specific token URL if needed
TOKEN_HEADERS = {}  # or your specific headers if needed

def keep_alive():
    if KEEPALIVE_URL:
        try:
            response = requests.get(KEEPALIVE_URL)
            if response.status_code == 200:
                print("Keep-alive successful")
            else:
                print("Keep-alive failed with status code:", response.status_code)
        except requests.RequestException as e:
            print("An error occurred:", e)

def get_token():
    if TOKEN_URL:
        try:
            response = requests.get(TOKEN_URL, headers=TOKEN_HEADERS)
            if response.status_code == 200:
                token = response.json().get('token')
                print("Token retrieved successfully:", token)
                return token
            else:
                print("Failed to retrieve token with status code:", response.status_code)
        except requests.RequestException as e:
            print("An error occurred:", e)
    return None

while True:
    keep_alive()
    token = get_token()
    time.sleep(INTERVAL)


#################################################################


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable
