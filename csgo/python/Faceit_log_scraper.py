import requests
import gzip
import os, json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
env_path = '../.env'
data_path = './logs/data'
demos_path = './logs/demos'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv('FACEIT_API_KEY')
FACEIT_API = "https://open.faceit.com/data/v4"
headers = {"Authorization": "Bearer %s"%API_KEY}
hub_ids = {
    "MythicDiamond": "56ac8cbe-a2b2-4630-8c6c-7e06ba0cd620",
    "MythicGold": "7e501b80-5f9d-4221-93d1-687f4cb07c12",
    "MythicSilver": "e5c0f56b-8bd4-4ad1-a9fe-524e88b2477f",
    "MythicBronze": "c6652c5a-ea24-4ca6-a2fe-211be287c139",
}


def main():
    for i in tqdm(range(0,5000, 100)):
        requestString = '%s/%s/%s/%s?%s=%d&%s=%d'%(FACEIT_API, 'hubs', hub_ids['MythicBronze'], 'matches', 'offset', i, 'limit', 100)
        print(requestString)
        JSONmatches = requests.get(requestString, headers=headers).content
        matches = json.loads(JSONmatches)['items']
        for match in matches:
            if match['status'] != 'FINISHED':
                continue
            else:
                matchId = match['match_id']
                demoUrl = match['demo_url'][0]
                with open('%s/%s.json'%(data_path, matchId), 'w') as data_file:
                    json.dump(match, data_file, indent=4, sort_keys=True)

                response = requests.get(demoUrl)
                if int(response.status_code) == 200:
                    with open('tempFile.gz', 'wb') as tempZip:
                        tempZip.write(response.content)
                    unZipped = gzip.open('tempFile.gz')
                    with open('%s/%s.dem'%(demos_path, matchId), 'wb') as out:
                        for line in unZipped:
                            out.write(line)

                
                

if __name__ == "__main__":
    main()