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
    
#-------------------------------------multiprocessing.Pool类的实例------------------------------#
import time
from multiprocessing import Pool
def run(fn):
  #fn: 函数参数是数据列表的一个元素
  time.sleep(1)
  return fn*fn

if __name__ == "__main__":
  testFL = [1,2,3,4,5,6]  
  print 'shunxu:' #顺序执行(也就是串行执行，单进程)
  s = time.time()
  for fn in testFL:
    run(fn)

  e1 = time.time()
  print "顺序执行时间：", int(e1 - s)

  print 'concurrent:' #创建多个进程，并行执行
  pool = Pool(5)  #创建拥有5个进程数量的进程池
  #testFL:要处理的数据列表，run：处理testFL列表中数据的函数
  rl =pool.map(run, testFL) 
  pool.close()#关闭进程池，不再接受新的进程
  pool.join()#主进程阻塞等待子进程的退出
  e2 = time.time()
  print "并行执行时间：", int(e2-e1)
  print rl
