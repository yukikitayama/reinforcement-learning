"""
https://www.geeksforgeeks.org/multiprocessing-python-set-1/
https://www.geeksforgeeks.org/multiprocessing-python-set-2/
https://www.geeksforgeeks.org/synchronization-pooling-processes-python/
"""

from multiprocessing import Pool, Process, Array, Value, Manager, \
    Queue, Pipe, Lock, Pool
import os
import time
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x * x


def print_square(num):
    print(f'Square: {num * num}')


def print_cude(num):
    print(f'Cude: {num * num * num}')


def worker1():
    print(f'ID of process running worker1: {os.getpid()}')


def worker2():
    print(f'ID of process running worker2: {os.getpid()}')


result = []


def square_list(mylist):
    global result

    for num in mylist:
        result.append(num)

    print(f'Result(in process p1): {result}')


def square_list_2(mylist, result, square_sum):
    for idx, num in enumerate(mylist):
        result[idx] = num * num

    square_sum.value = sum(result)

    # Needs [:] otherwise it outputs object name and memory key
    print(f'Result(in process p1): {result[:]}')

    print(f'Sum of squares(in process p1): {square_sum.value}')


def print_records(records):
    for record in records:
        print(f'Name: {record[0]}\nScore: {record[1]}\n')


def insert_record(record, records):
    records.append(record)
    print(f'New record added\n')


def square_list_3(mylist, q):
    for num in mylist:
        q.put(num * num)


def print_queue(q):
    print('Queue elements:')
    while not q.empty():
        print(q.get())
    print('Queue is now empty')


def sender(conn, msgs):
    for msg in msgs:
        conn.send(msg)
        print(f'Sent the message: {msg}')
    conn.close()


def receiver(conn):
    while 1:
        msg = conn.recv()
        if msg == 'END':
            break
        print(f'Received the message: {msg}')


def withdraw(balance):
    for _ in range(10000):
        balance.value = balance.value - 1


def deposit(balance):
    for _ in range(10000):
        balance.value = balance.value + 1


def perform_transactions():
    # balance is the shared data for concurrent access of process
    balance = Value('i', 100)

    p1 = Process(target=withdraw, args=(balance, ))
    p2 = Process(target=deposit, args=(balance, ))

    # This makes race condition, unpredictability
    p1.start()
    p2.start()

    # Wait until processes are finished
    p1.join()
    p2.join()

    print(f'Final balance = {balance.value}')


def withdraw_2(balance, lock):
    for _ in range(1000):
        # As soon as a lock is acquired, no other process can access its critical section until the lock is released.
        lock.acquire()
        balance.value = balance.value - 1
        lock.release()


def deposit_2(balance, lock):
    for i in range(1000):
        lock.acquire()
        balance.value = balance.value + 1
        lock.release()

        if i % 100 == 0:
            print(f'Current balance: {balance.value}')


def perform_transactions_2():
    balance = Value('i', 100)

    # A lock object
    lock = Lock()

    p1 = Process(target=withdraw_2, args=(balance, lock))
    p2 = Process(target=deposit_2, args=(balance, lock))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(f'Final balance = {balance.value}')


def square(n):
    # print(f'Worker process id for {n} : {os.getpid()}')
    return n * n


if __name__ == '__main__':

    # with Pool(5) as p:
    #     print(p.map(f, [1, 2, 3]))

    p1 = Process(target=print_square, args=(10, ))
    p2 = Process(target=print_cude, args=(10, ))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print('Done!')
    print('**********')
    print()

    print(f'ID of main process: {os.getpid()}')

    p1 = Process(target=worker1)
    p2 = Process(target=worker2)

    p1.start()
    p2.start()

    print(f'ID of process p1: {p1.pid}')
    print(f'ID of process p2: {p2.pid}')

    p1.join()
    p2.join()

    print('Both processes finished execution')

    print(f'Process p1 is alive: {p1.is_alive()}')
    print(f'Process p2 is alive: {p2.is_alive()}')
    print('**********')
    print()

    mylist = [1, 2, 3, 4]

    p1 = Process(target=square_list, args=(mylist, ))

    print('Start p1')
    p1.start()

    p1.join()
    print('End p1')

    print(f'Result(in main process): {result}')
    print('**********')
    print()

    # Create Array of int data type with space for 4 integers
    result = Array('i', 4)

    # Create Value of int data type
    square_sum = Value('i')

    p1 = Process(target=square_list_2, args=(mylist, result, square_sum))

    p1.start()

    p1.join()

    print(f'Result(in main program): {result[:]}')

    print(f'Sum of squares(in main program): {square_sum.value}')
    print('**********')
    print()

    # All the lines under the with statement block are under the scope of manager object
    with Manager() as manager:
        # records is created in server process memory
        records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin', 9)])
        new_record = ('Jeff', 8)

        p1 = Process(target=insert_record, args=(new_record, records))
        p2 = Process(target=print_records, args=(records, ))

        p1.start()
        p1.join()

        p2.start()
        p2.join()

    print('**********')
    print()

    q = Queue()

    p1 = Process(target=square_list_3, args=(mylist, q))
    p2 = Process(target=print_queue, args=(q, ))

    p1.start()
    p1.join()

    p2.start()
    p2.join()

    print('**********')
    print()

    msgs = ['hello', 'hey', 'hru?', 'END']

    # Pipe returns two connection objects for the two ends of the pipe
    parent_conn, child_conn = Pipe()

    # Message is sent from one end of pipe to another
    p1 = Process(target=sender, args=(parent_conn, msgs))
    # Receive messages at one end of a pipe
    p2 = Process(target=receiver, args=(child_conn, ))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print('**********')
    print()

    for _ in range(10):

        perform_transactions()

    print('**********')
    print()

    for _ in range(10):

        perform_transactions_2()

    print('**********')
    print()

    mylist = [i for i in range(100)]

    start_time = time.time()
    # The task is offloaded and distributed among the cores and processes automatically by Pool object
    # So we don't need to worry about creating processes explicitly
    p = Pool()
    # The contents of mylist and definition of square will be distributed among the cores
    result = p.map(square, mylist)
    # print(result)
    print(time.time() - start_time)

    start_time = time.time()
    result = []
    for l in mylist:
        result.append(square(l))
        # print(f'Worder process id for {l} : {os.getpid()}')
    # print(result)
    print(time.time() - start_time)

    print('**********')
    print()

    nums = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    result_single = []
    result_parallel = []

    for num in nums:
        print('num', num)
        mylist = [i for i in range(num)]

        # Single process
        start_time = time.time()
        squared_list = []
        for l in mylist:
            squared_list.append(square(l))
        result_single.append(time.time() - start_time)

        # Parallel processes
        start_time = time.time()
        p = Pool()
        squared_list = p.map(square, mylist)
        result_parallel.append(time.time() - start_time)

    plt.plot(result_single, label='Single')
    plt.plot(result_parallel, label='Parallel')
    plt.xticks(ticks=range(len(nums)), labels=nums, rotation=45)
    plt.title('Check performance of parallel processes')
    plt.xlabel('Number of squaring numbers')
    plt.ylabel('Time in second')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
