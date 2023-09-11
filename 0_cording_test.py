#이것이 취업음 위한 코딩테스트다 2021

#11강 파이썬 문법 - 자주 사용되는 표준 라이브러리
#
# itertools = 반복되는 형태 데이터 처리 유용 // 순령과 조합 라이브러리
# heapq = 힙 자료구조 // 우선순위 큐 기능 구현
# bisect = 이진탐색 기능 제공
# collections = 덱 ,카운터 등의 자료구조 포함
# math = 필수적인 수학

#10강 파이썬 문법 - 함수와 람다 표현식
"""
# def 함수명()매개변수:
#     실행할 소스코드
#     return 반환 값

def add(a,b):
    return a + b
print(add(3,7)) #함수 사용

#return을 안 쓸경우
def add2(c,d):
    print("함수의 결과",c + d)
add2(3,7)

# add2(d=3,c=7) <--파라미터릐 변수를 직접 지정 하면 순서 상관(X)
#global 키워드 <--함수 바깥에 존재하는 변수를 사용 할 수 있음
a= 0

def func():
    global a #리스트는 상관없음 #우선순위는 지역변수 -> 전역변수
    a += 1

for i in range(10):
    func()

    print(a)

#파이썬에서의 함수는 여러개의 반환 값을 가질 수 있음
#람다 표현식
#특정한 기능을 가지는 함수를 한줄에 작성 할 수 있음

print(lambda a,b: a+b(3,7)) #<<-- 덧셈 함수를 간단히 생성
#ex) sorted 등에 쓰임
array = [('홍길동',50),('이순신',32),('아무개',74)]

def my_key(x):
    return x[1]
print(sorted(array,key=my_key))
print(sorted(array,key=lambda x : x[1]))

#람다 리스트 예시
list1 = [1,2,3,4,5]
list2 = [6,7,8,9,10]

result = map(lambda a,b : a + b, list1,list2)
print(list(result))
"""

#9강 파이썬 문법 - 반복문
"""
i = 1
result = 0

#i가 9보다 작거나 같을 때 아래코드를 반복적으로 실행
while i <= 9:
    if i % 2 == 1:
        result += i
    i += 1
print(result)

#for문
#for 변수 in 리스트:
    #실행할 소스코드

array = [9,8,7,6,5] #리스트 말고 튜플도 가능

for x in array:
    print(x)

#for문 연속적인 값을 차례대로 순회할 때는 range()를 주로 사용
#이때 range(시작 값, 끝 값 + 1) 형태로 사용 / 인자를 하나만 넣으면 자동으로 시작 값은 0이됨
#예제
result = 0
# i는 1~9까지의 모든 값을 순회
for i in range(1,10): #시작값 : 1 / 끝값 : 9
    result += i
print(result)

#1 부터 30까지 모든 정수의 합 구하기
result = 0 # result 초기화
for i in range(1,31):
    result += i
print(result)

#continue 키워드 / 실행하지 말고 건너뛰어라!
# 1 부터 9 까지 모든 홀수의 합 구하기
result = 0
for i in range(1,10):
    if i % 2 == 0:
        continue
    result += i
print(result)

i = 1

while True:
    print("현재 i의 값:{0}".format(i))
    if i == 5:
        break
    i += 1

# 예제) 점수가 80점만 넘으면 합격 // 나중에 다시 해볼것
scores  = [80,92,99,14,77]
for i in range(5):
    if scores[i] >= 80:
        print("{0}번 학생은 합격 입니다.".format(i+1))
        
# 예제) 특정 번호의 학생은 제외하기
scores  = [80,92,99,14,77]
cheating_Student_list = {2,4}
for i in range(5):
    if i + 1 in cheating_Student_list: #해당번호 학생은 패스
        continue
    if scores[i] >= 80:
        print("{0}번 학생은 합격 입니다.".format(i+1))
#중첩된 반목문
#예제) 구구단 출력
for i in range(1,10):
    for j in range(1,10):
        print("{0} * {1} = {2}".format(i,j,i*j))
"""

#8강 파이썬 문법 - 조건문
"""
#if 문을 활용시 블록단위 잘 판단하기 이때 들여쓰기는 공백문자 4개를 입력함

#if : -> elif : -> else :
a = 5
if a >= 0:
    print("a>=0")
elif a >= -10:
    print("0 > a >= -10")
else:
    print("- 10> a ")

#파이썬의 기타 연산자
# x in 리스트 : 리스트 안에 x 가 있을 때 참이다.
# x not in 문자열 : 문자열 안에 x가 들어가 있지 않을 때 참이다.
# pass 키워드 :pass부븐 은 말 그대로 넘겨 버림

#조건문의 간소화
#줄 바꿈 안함 버전
score = 85
if score >= 80 : result = "Success"
else: result = "Fail"

#조건부 표현식 if~else문을 한 줄에 작성 할 수 있도록 해줍니다.
score = 85
result = "Success" if score >= 80 else "Fail"
print(result)
"""

#7강 파이썬 문법 - 기본 입출력
"""
# input() : 한 줄의 문자열을 입력 받는 함수임
# map() : 리스트의 모든 원소에 각각 특정한 함수를 적용할 때 사용함
# ex) list(map(int,input().split())) // 공백을 기준으로 구분된 데이터 입력받을 때 사용
# ex) a,b,c = (map(int,input().split())) //공백을 기준으로 구분된 데이터의 개수가 많지 않다면  사용

n = int(input()) #입력값을 정수형으로 입력을 받음
# data = input() # 이때 입력받은 결과는 하나의 문자열형태 이것을 공백기준으로 쪼개야함
data = list(map(int,input().split()))
#if 데이터가 반드시 3개 들어온다면?
#a,b,c = list(map(int,input().split()))
#print (a,b,c)

print(n)
print(data)

#입력을 최대한 빠르게 받아야 하는 경우
# input 대신 --> sys.stdin.readline() 을 사용
#sys.stdin.readline() 특징 = 한줄씩 입력받음 / 여러번 반복해야 하는 반복문에서 빠르게 사용 / 이진탐색,정렬,그래프관련문제 등 자주사용됨
# #!!!단, 입력 후 엔터가 줄 바꿈 기호로 입력 되므로 rstrip() 메서드를 함께 사용함

import sys
data = sys.stdin.readline().rstrip()
print(data)

#표준출력방법
#print() 사용 "," 사용시 띄어쓰기 가능 / 줄바꿈 원치 않을시 end 사용
a = 1
b = 2
print(a,b)
print(7, end=" ")
print(8, end=" ")

answer = 7
print("정답은" + str(answer) + "입니다.")

"""

#6강 파이썬 문법 - 사전,집합 자료형
"""
#집합 자료형
#특징 : 중복 허용 X, 순서 X, 리스트 or 문자열을 사용해서 초기화 가능함 --> 이때 set()함수를 사용함

#집합 자료형 초기화 방법 1
data = set([1,1,2,3,4,4,5])
print(data)
#집합 자료형 초기화 방법 2
data = {1,1,2,3,4,4,5}
print(data)
#두 개 모두 결과를 보면 중복을 허용 하지 않기 때문에 중복되는 원소는 합쳐지는 것을 알 수 있다.

#집합 자료형 연산
#ex) 두 집합 a,b
#합집합 (a : b) , 교집합 (a & b), 차집합 (a - b)

data = set([1,2,3])
print(data)

#새로운 원소를 추가
data.add(4)
print(data)

#새로운 원소를 여러개 추가
data.update([5,6])
print(data)

#특정한 값을 갖는 원소 삭제
data.remove(3)
print(data)


#사전자료형? --> 키와 값을 쌍을 데이터로 가지는 자료형
#키와 값을 별도로 뽑아 내기 위한 메서드를 지원함
# 키 데이터 추출 --> keys() , 값 데이터 추출 --> values()

data = dict()
data['사과'] = 'apple'
data['바나나'] = 'Banana'
data['코코넛'] = 'Coconut'

print(data)

if '사과' in data:
    print("'사과'를 키로 가지는 데이터가 존재합니다.")

# 키 데이터를 담은 리스트
key_list = data.keys()
# 값 데이터만 담은 리스트
value_list = data.values()
print(key_list)
print(value_list)


#각 키에 따른 값을 하나씩 출력
for key in key_list:
    print(data[key])
"""

#5강 파이썬 문법 - 문자열,튜플 자료형
"""
#튜플 = 한번 선언된 값을 변경 할 수 없음 , () 소괄호 사용 , 적은 메모리 사용
#튜플 사용 장점 : 서로 다른 성질의 데이터를 묶어서 관리 할 때 , 튜플은 변경이 불가능 하기때문에 키 값으로 사용 할 수 있음
a = (1,2,3,4,5,6,7,8,9)
print(a[3])


#문자열 연산
a  ="hello"
b = "world"
print(a+b)

a = "String" #문자열 연산도 가능
print(a * 3)

a = "ABCDEF"
print(a[2:4]) #index2~4 까지 가져와라!
"""

#4강 파이썬 문법 - 리스트 자료형
"""
#리스트는 대괄호[] 안에 원소를 넣어 초기화, 쉼표(,)로 구별
#!!인덱스는 0부터 시작 ex)0,1,2,3...

#직접 데이터를 넣어 초기화
a = [1,2,3,4,5,6,7,8,9]
print(a)

#네 번째 원소만 출력
print(a[3])

a[4] = 4 #4번째 인덱스를 4로 수정
print(a)

#뒤에서 세 번째 원소 출력 "-" 는 거꾸로 탐색 
print(a[-3])

#크기가 N이고, 모든 값이 0인 1차원 리스트 초기화
n = 10 #n개 리스트에
a = [0] * n # 모든 인덱스에 0을 넣어 초기화
print(a)
"""

# 3강 파이썬 문법 - 수 자료형
"""
#정답은 0.9가 맞지만 이진수로는 완벽하게 0.9표현이 안됨

a = 0.3 + 0.6
# print(a)
print(round(a,4))

if a == 0.9:
    print(True)
else:
    print(False)
# 따라서 0.89999...가 나오면서 False가 나옴
#해결방안 ! round(해당값,반올림자릿수) 함수를 이용하면 됨

# 나누기 연산자 "/" 를 주의해서 사용할 것
#why? 파이썬에서는 나눠진 결과를 실수형으로 반환하기 때문
#나머지 연산자 "%"  ex) a가 홀수인지 체크해야하는 경우
"""