
import queue
import time
start_time = time.time()  # 시작 시간 저장
# 큐 객체 생성
my_queue = queue.Queue()
for i in range(0,10000):
# 아이템 추가
    my_queue.put(1)
    my_queue.put(2)
    my_queue.put(3)

# 아이템 삭제
    item = my_queue.get()
    print(item)  # 출력: 1

    item = my_queue.get()
    print(item)  # 출력: 2

    item = my_queue.get()
    print(item)  # 출력: 3

end_time = time.time()
print("run time :", end_time - start_time)  # 현재시각 - 시작시간 = 실행 시간
#run time : 0.18340730667114258

##############

import time
start_time = time.time()  # 시작 시간 저장
my_queue = []

# 아이템 추가
for i in range(0,10000):
    my_queue.append(1)
    my_queue.append(2)
    my_queue.append(3)

# 아이템 삭제
    item = my_queue.pop(0)
    print(item)  # 출력: 1

    item = my_queue.pop(0)
    print(item)  # 출력: 2

    item = my_queue.pop(0)
    print(item)  # 출력: 3
end_time = time.time()
print("run time :", end_time - start_time)  # 현재시각 - 시작시간 = 실행 시간
#run time : 0.18340730667114258
#run time : 0.09856081008911133
