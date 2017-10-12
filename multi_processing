import multiprocessing
import concurrent
from tqdm import tqdm
# multiprocessing.cpu_count()

def task(path):
  pass
  
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(task, arg) for arg in targets}
    for f in tqdm(concurrent.futures.as_completed(futures),total=len(targets)):
        pass
  
