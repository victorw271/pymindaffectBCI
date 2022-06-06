import datetime
from pytz import timezone
import git
from pygit2 import Repository
import csv
import os

dir = None

def get_metadata():
    current_date_and_time = datetime.datetime.now()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch = Repository('.').head.shorthand # branch
    return current_date_and_time, sha, branch

def meta2csv():
    current_date_and_time, sha, branch = get_metadata()
    os.mkdir('csv/' + sha)
    with open('csv/'+ sha +'/metadata.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['current_date_and_time', 'sha', 'branch'])
        writer.writerow([current_date_and_time, sha, branch])
    return sha

 