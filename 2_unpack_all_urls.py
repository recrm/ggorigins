from joblib import delayed, Parallel

import requests
from joblib import Memory
import json

# Modify this to change where caching saves to.
# Super important as this is a long process.
memory = Memory(location="/tmp")


@memory.cache(verbose=0)
def unpack(url):
    try:
        response = requests.get(url, timeout=2)
    except Exception as e:
        return e.request.url

    return response.url


if __name__ == "__main__":
    """This is the script I used to unpack all the t.co links. The all_urls.json file is generated in the clean_data 
    script."""
    with open("output/all_urls.json") as f:
        data = json.load(f)

    with Parallel(n_jobs=-1, verbose=5) as p:
        f = delayed(unpack)
        gen = (f(i) for i in data)
        results = p(gen)

    final = {k: v for k, v in zip(data, results)}

    with open("gamergate/url_dictionary.json", "w+") as f:
        json.dump(final, f)
