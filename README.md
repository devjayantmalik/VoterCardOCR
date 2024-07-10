# First of all install

```shell
pip install "python-doctr[torch]"
```

## Files and their purpose

| Filename    | Description                                                                                                             | Command             |
|-------------|-------------------------------------------------------------------------------------------------------------------------|---------------------|
| main.py     | Run this file for getting data from api and performing inference                                                        | python3 main.py     |
| build_db.py | This files scans all pdf from data/documents directory and prepares a sqlite database for api to work                   | python3 build_db.py |
| api.py      | This is the api servers used by inference agents implemented in main.py file. Exposes fastapi endpoints for all clients | python3 api.py      |
| stats.py | This file performs database queries and tells us status of what's going on. | python3 stats.py|
