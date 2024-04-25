from collections import deque
import random
import time
# given a dots and boxes board state return the best action

class Edge:
    """
    Represents an edge in the board of game.

    Attributes:
        is_colored (bool) : shows if an edge is colored or not.
        is_peripheral (bool) : shows if an edge is located at the marginal position.
        is_vertical (bool) : shows whether an edge is vertical.
        is_horizontal (bool) : shows whether an edge is horizontal.

    """

    def __init__(self, a: tuple, b: tuple):
        """
        Args:
            a (tuple) : represents starting cord of an edge.
            b (tuple) : represents ending cord of an edge.
        """
        self.cord = (a, b)
        self.is_colored = False
        self.is_peripheral = False
        self.is_vertical = False
        self.is_horizontal = False
        # adjust the ordering of coordinates for simplification -> starting point of an edge is
        # above the ending point or at the left of it.
        if self.cord[0][0] > self.cord[1][0]:
            self.cord = ((self.cord[1]), (self.cord[0]))
        if self.cord[0][1] > self.cord[1][1]:
            self.cord = ((self.cord[1]), (self.cord[0]))

    def color(self):
        """
        sets `is_colored` to True.
        Invoked when a move is applied on an edge.

        See Also:
            `Board.move()`
        """
        self.is_colored = True

    def edge_type(self, n, m):
        """
        determines if type of edge when invoked.
        An edge is vertical or horizontal and also could be located at the margin of a board.
        :param n: size of the game board(rows).
        :param m: size of the game board(cols).
        :return:
        """
        if self.cord[0][1] == self.cord[1][1]:
            self.is_vertical = True
            if self.cord[0][1] == m or self.cord[0][1] == 0:
                self.is_peripheral = True
        else:
            self.is_horizontal = True
            if self.cord[0][0] == n or self.cord[0][0] == 0:
                self.is_peripheral = True

    def __eq__(self, other):
        """
        Two edges are considered equal if share identical coordinates.
        :param other: the edge to be compared with.
        :return: whether they are identical.
        """
        if not isinstance(other, Edge):
            return False
        return self.cord == other.cord


class Box:
    def __init__(self, x, y, colored_edges: tuple):
        self.coordinates = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
        self.loc = (x, y)
        self.topLine = Edge(self.coordinates[0], self.coordinates[2])
        self.rightLine = Edge(self.coordinates[2], self.coordinates[3])
        self.bottomLine = Edge(self.coordinates[1], self.coordinates[3])
        self.leftLine = Edge(self.coordinates[0], self.coordinates[1])
        self.lines = [self.topLine, self.rightLine, self.bottomLine, self.leftLine]
        self.colored_edges = colored_edges
        self.color_surroundings()
        self.owner = None
        self.complete = False

    def color_surroundings(self):
        # i = 0
        for colored in self.colored_edges:
            for i, surrounding in enumerate(self.lines):
                if colored == surrounding.cord:
                    self.lines[i].is_colored = True
                    i += 1
                    break

    def colorBoxEdge(self, edge: Edge, un_make=False):
        if edge in self.lines:
            if edge == self.topLine:
                if un_make:
                    self.topLine.is_colored = False  # unmake move
                else:
                    self.topLine.is_colored = True
            elif edge == self.rightLine:
                if un_make:
                    self.rightLine.is_colored = False  # unmake move
                else:
                    self.rightLine.is_colored = True
            elif edge == self.bottomLine:
                if un_make:
                    self.bottomLine.is_colored = False  # unmake move
                else:
                    self.bottomLine.is_colored = True
            else:
                if un_make:
                    self.leftLine.is_colored = False  # unmake move
                else:
                    self.leftLine.is_colored = True
        if (self.topLine.is_colored == True and self.rightLine.is_colored == True and
                self.bottomLine.is_colored == True and self.leftLine.is_colored == True):
            self.complete = True
        else:
            self.complete = False  # if unmaking a move removes captured box


class Board:
    def __init__(self, n, m, coloredEdges):
        self.n = n - 1
        self.m = m - 1
        self.myScore = 0
        self.opponentScore = 0
        coloredEdges = self.adjust_cord(coloredEdges)
        self.boxes = self.generateBoxes(coloredEdges)
        self.availableMoves = deque()  # will get instanced in generateEdges()
        self.edges = self.generateEdges(coloredEdges)
        self.alpha = -float('inf')
        self.beta = float('inf')

    def adjust_cord(self, colored_edges):
        temp = []
        for cord in colored_edges:
            if cord[0][0] > cord[1][0]:
                cord = ((cord[1]), (cord[0]))
                temp.append(cord)
            elif cord[0][1] > cord[1][1]:
                cord = ((cord[1]), (cord[0]))
                temp.append(cord)
            else:
                temp.append(cord)
        return temp

    def generateEdges(self, coloredEdges):
        '''
        generates all the edges and sets isColored True for already colored edges and
        instantiates availableMoves
        :param coloredEdges: already colored edges in given state
        :return: all edges of the board
        '''

        rows = self.n
        cols = self.m
        edges = deque()
        for i in range(rows + 1):
            for j in range(cols):
                edge = Edge((i, j), (i, j + 1))
                edge.edge_type(rows, cols)
                if edge.cord in coloredEdges:
                    edge.color()
                else:
                    self.availableMoves.append(edge)
                edges.append(edge)
        for i in range(rows):
            for j in range(cols + 1):
                edge = Edge((i, j), (i + 1, j))
                edge.edge_type(rows, cols)
                if edge.cord in coloredEdges:
                    edge.color()
                else:
                    self.availableMoves.append(edge)
                edges.append(edge)
        return edges

    def generateBoxes(self, colored_edges):
        boxes = [[Box(j, i, colored_edges) for i in range(self.m)] for j in range(self.n)]
        return boxes

    def _boxes_for_vertical(self, i, j, edge):
        covering_boxes = []
        if edge.is_peripheral:
            if edge.cord[0][1] == 0:
                covering_boxes.append(self.boxes[i][j])
            else:
                covering_boxes.append(self.boxes[i][j - 1])
        else:
            covering_boxes.append(self.boxes[i][j])
            covering_boxes.append(self.boxes[i][j - 1])
        return covering_boxes

    def _boxes_for_horizontal(self, i, j, edge):
        covering_boxes = []
        if edge.is_peripheral:
            if edge.cord[0][0] == 0:
                covering_boxes.append(self.boxes[i][j])
            else:
                covering_boxes.append(self.boxes[i - 1][j])
        else:
            covering_boxes.append(self.boxes[i][j])
            covering_boxes.append(self.boxes[i - 1][j])
        return covering_boxes



    def move(self, action: Edge, turn: int):
        """
        makes the move on board
        :param action:
        :param captured: indicates if the move resulted in capturing a box
        :param turn: 1 is max 0 is min ,max is me
        :return:
        """
        for edge in self.availableMoves:
            if edge == action:
                edge.color()
                break
        self.availableMoves.remove(action)
        covering_boxes = []
        i = action.cord[0][0]
        j = action.cord[0][1]

        if edge.is_vertical:
            covering_boxes = self._boxes_for_vertical(i, j, edge)
        elif edge.is_horizontal:
            covering_boxes = self._boxes_for_horizontal(i, j, edge)

        for box in covering_boxes:
            box.colorBoxEdge(action)
            if box.complete and box.owner == None:
                box.owner = turn
                # print("box captured")
                if turn == 1:
                    self.myScore += 1
                else:
                    self.opponentScore += 1
                return True
        return False
    def unmake_move(self, action: Edge):
        for edge in self.edges:
            if edge == action:
                edge.is_colored = False
                break
        self.availableMoves.append(action)
        covering_boxes = []
        i = action.cord[0][0]
        j = action.cord[0][1]
        if edge.is_vertical:
            covering_boxes = self._boxes_for_vertical(i, j, edge)
        elif edge.is_horizontal:
            covering_boxes = self._boxes_for_horizontal(i, j, edge)

        for box in covering_boxes:
            box_was_complete = box.complete
            prev_box_owner = box.owner
            box.colorBoxEdge(action, un_make=True)
            if box_was_complete != box.complete:  # the removed edge decaptured the box
                box.owner = None
                if prev_box_owner == 1:
                    self.myScore -= 1
                else:
                    self.opponentScore -= 1


class Ai:
    def __init__(self, shape):
        self.shape = shape
        self.X = shape[0]
        self.Y = shape[1]

    # state = [((x0, y0),(x1, y1)), ...]
    def decide(self, state):
        board = Board(self.X, self.Y, state)
        move: Edge
        depth = self.adjust_depth(board)
        print("depth is " + str(depth))
        _, move = self.minimax(board, board.availableMoves, depth, 1)
        if move:
            return move.cord
        return None

    def adjust_depth(self, state: Board):
        available_moves = len(state.availableMoves)
        print(available_moves)
        if self.X == 5:
            return 4
        threshold = 0
        if self.X == 10:
            threshold = 50
            if available_moves > threshold:
                return 2
            elif available_moves > threshold - 15:
                return 3
            else:
                return 4
        else:
            threshold = 90
            if available_moves > threshold:
                return 2
            elif available_moves > threshold - 15:
                return 3
            else:
                return 4

    def minimax(self, state: Board, available_moves: deque, depth, isMax, alpha=None, beta=None):
        if alpha is None:
            alpha = state.alpha
        if beta is None:
            beta = state.beta
        if depth == 0 or len(available_moves) == 0:
            return self.evaluateLeaf(state), None
        if isMax:
            best_move = None
            max_eval = -float('inf')
            for move in list(available_moves):
                # successor = deepcopy(state)  # OPTIMIZE
                # is_box_captured = successor.move(move, 1)
                is_box_captured = state.move(move, 1)
                turn = 1 if is_box_captured else 0
                # eval, _ = self.minimax(successor, successor.availableMoves, depth - 1, turn, alpha, beta)
                eval, _ = self.minimax(state, state.availableMoves, depth - 1, turn, alpha, beta)
                state.unmake_move(move)  # NEW
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in list(available_moves):
                # successor = deepcopy(state)  # OPTIMIZE
                # is_box_captured = successor.move(move, 0)
                is_box_captured = state.move(move, 0)
                turn = 0 if is_box_captured else 1
                # eval, _ = self.minimax(successor, successor.availableMoves, depth - 1, turn, alpha, beta)
                eval, _ = self.minimax(state, state.availableMoves, depth - 1, turn, alpha, beta)
                state.unmake_move(move)  # NEW
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluateLeaf(self, state: Board):
        return state.myScore - state.opponentScore


@staticmethod
def create_random_board(shape):
    lines = []
    maximum_lines = (shape[0] + 1) * shape[1] + (shape[1] + 1) * shape[0]
    number_of_lines = random.randint(1, maximum_lines)
    n = 0
    while n < number_of_lines:
        points = []
        first = (random.randint(0, shape[0]), random.randint(0, shape[1]))
        if random.randint(0, 1) == 1:
            second = (first[0], first[1] + 1)
        else:
            second = (first[0] + 1, first[1])
        if second[0] > shape[0] or second[1] > shape[1]:
            continue
        points.append(first)
        points.append(second)
        points.sort()
        line = tuple(points)
        if line in lines:
            continue
        else:
            lines.append(line)
            n += 1

    lines.sort()
    return shape, lines

def main():
    avg = 0
    timeout = 0
    n = 50
    size = 8
    random.seed(10)
    for x in range(n):
        shape, state = create_random_board((size, size))
        #print(state)
        start_time = time.time()
        ai = Ai(shape)
        move = ai.decide(state)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if (elapsed_time > 4.8):
            print(f"{elapsed_time} seconds to complete.")
            timeout += 1
        avg += elapsed_time
        print(move)
        print("____________________________")
    print("avg is: " + str(avg / n))
    print("timeout's count: " + str(timeout))


if __name__ == '__main__':
    main()
