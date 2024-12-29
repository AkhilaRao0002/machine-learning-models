#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
from heapq import heappush, heappop

class PuzzleNode:
    def __init__(self, board, empty_pos, moves=0, parent=None):
        self.board = board
        self.empty_pos = empty_pos
        self.moves = moves
        self.parent = parent
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        #count misplaced tiles(except empty)
        count=0
        goal = [[1,2,3],[4,5,6],[7,8,0]]
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0 and self.board[i][j] != goal[i][j]:
                    count +=1
        return count + self.moves

    def __lt__(self, other):
        return self.cost < other.cost

def get_possible_moves(pos):
    moves = []
    row, col = pos
    for dr, dc in [(-1,0), (1,0), (0, -1), (0,1)]:
        new_row, new_col = row+dr, col+dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            moves.append((new_row, new_col))
    return moves

def solve_puzzle(initial_board):
    empty_pos = None
    for i in range(3):
        for j in range(3):
            if initial_board[i][j] == 0:
                empty_pos = (i,j)
               # print(empty_pos)
                break

    start_node = PuzzleNode(initial_board, empty_pos)
    pq = []
    heappush(pq, start_node)
    seen = set()

    while pq:
        current = heappop(pq)
        if current.cost - current.moves ==0:
            path = []
            while current:
                path.append(current.board)
                current = current.parent
            return path[::-1]

        for new_pos in get_possible_moves(current.empty_pos):
            new_board = copy.deepcopy(current.board)
            r1, c1 = current.empty_pos
            r2, c2 = new_pos
            new_board[r1][c1], new_board[r2][c2] = new_board[r2][c2], new_board[r1][c1]

            board_tuple = tuple(map(tuple, new_board))
            if board_tuple not in seen:
                seen.add(board_tuple)
                new_node = PuzzleNode(new_board, new_pos, current.moves +1, current)
                heappush(pq, new_node)
    return None

initial = [
    [4, 3, 1],
    [2, 5, 6],
    [8, 7, 0]
]

solution = solve_puzzle(initial)
if solution:
    print("Solution found! steps:")
    for i, board in enumerate(solution):
        print(f"\nStep {i}:")
        for row in board:
            print(row)
else:
    print("No solution found")


# In[ ]:




