import multiprocessing
import concurrent
from tqdm import tqdm
# multiprocessing.cpu_count()

def task(path):
  pass
  
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(task, arg) for arg in targets}
#     for f in tqdm(concurrent.futures.as_completed(futures),total=len(targets)):
#         pass
    try:
        for f in tqdm(concurrent.futures.as_completed(futures),total=len(targets)):
            err = f.exception()
            if err is not None:
               raise err
    except KeyboardInterrupt:
        print("stopped by hand")
#-----------------------------------------------------------------------------------------#  
#coding=utf-8
import multiprocessing

def do(n) :
  #获取当前线程的名字
  name = multiprocessing.current_process().name
  print name,'starting'
  print "worker ", n
  return 

if __name__ == '__main__' :
  numList = []
  for i in xrange(5) :
    p = multiprocessing.Process(target=do, args=(i,))
    numList.append(p)
    p.start()
    p.join()
    print "Process end."
