import queue
import collections
import copy
import time
from collections import deque

legal_moves = {'d': [1, 0], 'r': [0, 1], 'u': [-1, 0],
               'l': [0, -1]}  # counter-clockwise starting from downside neighbor


class state():
    cord = None
    parent = None
    board = []
    path_to_curr = []

    def __init__(self, cord, parent, board=None, path_to_curr=None):
        self.cord = cord
        self.parent = parent
        # self.new_inside = new_inside
        self.board = board
        self.path_to_curr = path_to_curr


class My_board():
    board = []

    def __init__(self, board):
        self.board = board


# evaluate starting state-where our agent is located
def init_location(board):
    n = len(board)
    m = len(board[0])
    for i in range(n):
        for j in range(m):
            if (board[i][j]) == 'a':
                return [i, j]


# evaluate number of stones
def get_num_stones(board):
    s = 0
    n = len(board)
    m = len(board[0])
    for i in range(n):
        for j in range(m):
            if (board[i][j]) == 's':
                s += 1
    return s


# auxiliary function to check if cord is in board or not
def is_out_of_range(x, y, n, m):
    if (x < 0 or y < 0
            or x > n or y > m):
        return True
    return False


def is_in_corner(x, y, n, m):
    if (x == y == 0
            or (x == n and y == m)
            or (x == 0 and y == m)
            or (x == n and y == 0)):
        return True
    return False


def is_legal(cord, board, direction, deadlock_free_tiles):
    n = len(board) - 1
    m = len(board[0]) - 1
    x = cord[0]
    y = cord[1]
    # in range or not
    if is_out_of_range(x, y, n, m):
        return False
    # wall
    elif board[x][y] == "w":
        return False
    # moving a stone or occupied box
    elif board[x][y] == "s" or board[x][y] == "B":  # NEW
        move_x = x + legal_moves[direction][0]
        move_y = y + legal_moves[direction][1]
        # stone goes out of the board
        if is_out_of_range(move_x, move_y, n, m):
            return False
        in_direction_of_stone = board[move_x][move_y]
        # another stone exists in the direction of moving
        if in_direction_of_stone == "s":
            return False
        # a wall exists in the direction of moving
        elif in_direction_of_stone == "w":
            return False
        elif in_direction_of_stone == "B":  # NEW
            return False
        # deadlock - stone goes to a corner
        elif is_in_corner(move_x, move_y, n,m) and board[move_x][move_y] != "b":
            # print("deadlock prevented")
            return False
        # elif (move_x, move_y) not in deadlock_free_tiles:
        #     print("deadlock prevented")
        #     return False
    return True


# update the board with respect to the path generated and return it next to its number of s for goal tes
def update_board(path, board_org, init_cord):
    board = copy.deepcopy(board_org)
    num_stone_in_box = 0
    agent_x = init_cord[0]
    agent_y = init_cord[1]
    agent_new_x = agent_x
    agent_new_y = agent_y
    for move in path:
        agent_new_x += legal_moves[move][0]
        agent_new_y += legal_moves[move][1]
        new_inside = board[agent_new_x][agent_new_y]

        # moving to free space
        if new_inside == "f":
            board[agent_new_x][agent_new_y] = "a"
        # moving to EMPTY box
        elif new_inside == "b":
            board[agent_new_x][agent_new_y] = "ba"  # we show a tile with both agent and empty box as ba
        elif new_inside == "s" or new_inside == "B":  # NEW
            stone_x = agent_new_x + legal_moves[move][0]  # where is the stone moving to
            stone_y = agent_new_y + legal_moves[move][1]
            # moving the stone to free space
            if board[stone_x][stone_y] == "f":
                if new_inside == "B":  # NEW
                    board[agent_new_x][agent_new_y] = "ba"  # a B f -> f ba s
                    num_stone_in_box += -1  # removing a stone from a box
                else:
                    board[agent_new_x][agent_new_y] = "a"
                board[stone_x][stone_y] = "s"
                # moving the stone to box
            elif board[stone_x][stone_y] == "b":  # NEW
                if new_inside == "B":
                    board[agent_new_x][agent_new_y] = "ba"  # a B b -> f ba B
                else:
                    board[agent_new_x][agent_new_y] = "a"
                    num_stone_in_box += 1
                board[stone_x][stone_y] = "B"  # B demonstrates an occupied box  NEW
                # num_stone_in_box += 1
            # elif board[stone_x][stone_y] == "B":
            #     print("tada")

        if (board[agent_x][agent_y]) == "a":
            board[agent_x][agent_y] = "f"
        else:  # if we are leaving a tile with empty box aka 'ba'
            board[agent_x][
                agent_y] = "b"  # this else statement is unnecessary but is written to demonstrate the process

        agent_x = agent_new_x
        agent_y = agent_new_y

    return [copy.deepcopy(board), num_stone_in_box]


def heuristic(agent,stones,boxes):
    h = 0
    ds = []
    for stone in stones:
        if(stones and boxes):
            ds.append(min([manhattan(stone,box) for box in boxes]))
    if(ds):
        h =  sum(ds)
        h += sum([manhattan(agent,box) for box in boxes])
        h += min([manhattan(agent,stone) for stone in stones])
        return h

    return 0


def manhattan(x, goal):
    return abs(x[0] - goal[0]) + abs(x[1] - goal[1])


# detects simple deadlocks:
# for each box
# delete all stones from the board
# place a stone on the box
# PULL the box from the goal square to every possible
#  square and mark all reached squares as visited
# returns cords which are valid for a stone to be pushed in
def precalculate_deadlocks(board, stones, boxes):
    temp = copy.deepcopy(board)
    valid_tiles = set()

    visited = set()
    def can_reach(init_x,init_y,dest_x,dest_y):

        if is_out_of_range(init_x, init_y, len(board) - 1, len(board[0]) - 1) or (board[init_x][init_y] == "w") or ((init_x, init_y) in visited):
            return False
        visited.add((init_x, init_y))
        if (init_x == dest_x) and (init_y == dest_y):
            return True
        return (can_reach(init_x+legal_moves['r'][0],init_y+legal_moves['r'][1],dest_x,dest_y) or
                can_reach(init_x+legal_moves['l'][0],init_y+legal_moves['l'][1],dest_x,dest_y) or
                can_reach(init_x+legal_moves['d'][0],init_y+legal_moves['d'][1],dest_x,dest_y)or
                can_reach(init_x+legal_moves['u'][0],init_y+legal_moves['u'][1],dest_x,dest_y))



    for box_x, box_y in boxes:
        for stone_x, stone_y in stones:
            temp[stone_x][stone_y] = "f"
        temp[box_x][box_y] = "s"

        for i in range(len(board)):
            for j in range(len(board[0])):
                if (board[i][j]) == "w":
                    continue

                if can_reach(i,j,box_x,box_y):
                    valid_tiles.add((i,j))
            visited.clear()
    return valid_tiles


def stones_and_boxes_cord(board, stones, boxes):
    n = len(board)
    m = len(board[0])
    for i in range(n):
        for j in range(m):
            if board[i][j] == "s":
                stones.append([i, j])
            if board[i][j] == "b":
                boxes.append([i, j])


def bfs(board2D):
    board = My_board(copy.deepcopy(board2D))
    num_stones = get_num_stones(board.board)
    frontier = queue.Queue()  # in reached but not yet expanded
    reached = {}  # have been visited whether expanded or not
    init_cord = init_location(board.board)
    initial_state = state(cord=init_cord, parent=None, board=board2D, path_to_curr=[])
    reached[tuple(map(tuple, board2D))] = True  # start from the state which are agent is located in
    frontier.put(initial_state)

    # precalculating simple deadlocks cord
    stones = []
    boxes = []
    stones_and_boxes_cord(board2D, stones, boxes)
    deadlock_free_tiles = precalculate_deadlocks(board2D, stones, boxes)
    while (frontier.qsize() > 0):

        curr_state = frontier.get()  # remove the state to be expanded
        # add the neighbours of current state if a that move is legal
        for direction in legal_moves:
            # print(f'direction is: {direction}')
            neighbour_cord = [curr_state.cord[0] + legal_moves[direction][0],
                              curr_state.cord[1] + legal_moves[direction][1]]
            if is_legal(neighbour_cord, curr_state.board, direction, deadlock_free_tiles):
                pcopy = copy.deepcopy(curr_state.path_to_curr)
                pcopy.append(direction)
                neighbour = state(cord=neighbour_cord, parent=curr_state,
                                  path_to_curr=pcopy)
                achieved_state = update_board(neighbour.path_to_curr, board2D, init_cord)
                neighbour.board = achieved_state[0]
                if num_stones == achieved_state[1]:  # early goal test
                    # print("here???")
                    return neighbour.path_to_curr
                # print(achieved_state[0])
                if not tuple(map(tuple, achieved_state[0])) in reached:
                    reached[tuple(map(tuple, achieved_state[0]))] = True
                    frontier.put(neighbour)

                board.board = board2D  # reset the board


def dfs(board2D):
    board = My_board(copy.deepcopy(board2D))
    num_stones = get_num_stones(board.board)
    frontier = deque()  # in reached but not yet expanded
    reached = {}  # have been visited whether expanded or not
    init_cord = init_location(board.board)
    initial_state = state(cord=init_cord, parent=None, board=board2D, path_to_curr=[])
    reached[tuple(map(tuple, board2D))] = True
    # start from the state which are agent is located in
    frontier.append(initial_state)

    # precalculating simple deadlocks cord
    stones = []
    boxes = []
    stones_and_boxes_cord(board2D, stones, boxes)
    deadlock_free_tiles = precalculate_deadlocks(board2D, stones, boxes)
    while len(frontier) > 0:
        curr_state = frontier.pop()  # remove the state to be expanded

        # add the neighbours of current state if a that move is legal
        for direction in legal_moves:
            neighbour_cord = [curr_state.cord[0] + legal_moves[direction][0],
                              curr_state.cord[1] + legal_moves[direction][1]]
            if is_legal(neighbour_cord, curr_state.board, direction, deadlock_free_tiles):
                pcopy = copy.deepcopy(curr_state.path_to_curr)
                pcopy.append(direction)
                neighbour = state(cord=neighbour_cord, parent=curr_state,
                                  path_to_curr=pcopy)
                achieved_state = update_board(neighbour.path_to_curr, board2D, init_cord)
                neighbour.board = achieved_state[0]
                if num_stones == achieved_state[1]:  # early goal test
                    return neighbour.path_to_curr
                if not tuple(map(tuple, achieved_state[0])) in reached:
                    reached[tuple(map(tuple, achieved_state[0]))] = True
                    frontier.append(neighbour)

def a_star(board2D):
    board = My_board(copy.deepcopy(board2D))
    num_stones = get_num_stones(board.board)
    frontier = []  # in reached but not yet expanded
    reached = {}  # have been visited whether expanded or not
    fs = {}
    init_cord = init_location(board.board)
    init_cord = init_location(board.board)
    initial_state = state(cord=init_cord, parent=None, board=board2D, path_to_curr=[])
    reached[tuple(map(tuple, board2D))] = True
    # start from the state which are agent is located in
    frontier.append(initial_state)

    # precalculating simple deadlocks cord
    stones = []
    boxes = []
    stones_and_boxes_cord(board2D, stones, boxes)
    deadlock_free_tiles = precalculate_deadlocks(board2D, stones, boxes)
    print((len(board2D) * len(board2D[0])) - len(deadlock_free_tiles))
    for i in range(len(board2D) ):
        for j in range(len(board2D[0])):
            if (i,j) not  in deadlock_free_tiles:
                print(i,j)

    # f = g + h -> g is the number of moves to get to the current state
    fs.update({initial_state : (len(initial_state.path_to_curr) + heuristic(initial_state.cord,stones,boxes))})
    while len(frontier) > 0:
        curr_state = min(frontier,key= lambda node:fs[node])
        frontier.remove(curr_state)  # remove the state to be expanded
        fs.pop(curr_state)

        # add the neighbours of current state if a that move is legal
        for direction in legal_moves:
            neighbour_cord = [curr_state.cord[0] + legal_moves[direction][0],
                              curr_state.cord[1] + legal_moves[direction][1]]
            if is_legal(neighbour_cord, curr_state.board, direction, deadlock_free_tiles):
                pcopy = copy.deepcopy(curr_state.path_to_curr)
                pcopy.append(direction)
                neighbour = state(cord=neighbour_cord, parent=curr_state,
                                  path_to_curr=pcopy)
                achieved_state = update_board(neighbour.path_to_curr, board2D, init_cord)
                neighbour.board = achieved_state[0]

                # check the frozen deadlock here on neighbor
                if num_stones == achieved_state[1]:  # early goal test
                    return neighbour.path_to_curr
                if not tuple(map(tuple, achieved_state[0])) in reached:
                    reached[tuple(map(tuple, achieved_state[0]))] = True
                    frontier.append(neighbour)

                    neighbour_stones = []
                    neighbour_boxes = []
                    stones_and_boxes_cord(neighbour.board,neighbour_stones,neighbour_boxes)
                    fs.update({neighbour : len(neighbour.path_to_curr) + heuristic(neighbour.cord,neighbour_stones,neighbour_boxes)})

                board.board = board2D  # reset the board

    return False



def main():
    board =  [['a', 'f', 'f'],
             ['f', 's', 'b'],
             ['f', 'f', 'f'],
             ['f', 'f', 'f']]
    start_time = time.time()
    print(a_star(board))
    end_time = time.time()
    duration = end_time - start_time
    print(f"Function took {duration} seconds to run.")


if __name__ == "__main__":
    main()
