import queue
# only staisfies row and column wise constraints and not the subgrides

class SudokuCSP:
    solved_board = None

    def __init__(self, board):
        self.board = board
        self.variables = self.define_variables()
        self.domains = self.define_domains()
        self.constraints = self.define_constraints()

    def define_variables(self):
        """
        each tile in the board is a variable starting from (0,0) to (n-1,n-1)
        :return: list of variables
        """
        n = len(self.board)
        return [(i, j) for i in range(n) for j in range(n)]

    def define_domains(self):
        """
        for unassigned variables are [1,n]
        and for pre-filled ones is the value assigned

        :return: a dictionary - the keys are the variables and values are domain set
        """
        n = len(self.board)
        return {tile: set(range(1, n+1)) if self.board[tile[0]][tile[1]] == 0
            else {self.board[tile[0]][tile[1]]} for tile in self.variables}


    def define_constraints(self):
        """
        only row and columns constraints are considered
        since we have AllDiff constraint for each row and column we convert it to appropriate binary constraint
        :return: a dictionary which its keys are nodes and its values are its neighbors in constraint graph
        """
        constraints = {}
        n = len(self.board)
        for i in range(n):
            for j in range(n):
                constraints[(i,j)] = [(m,l) for m in range(n) for l in range(n) if (m,l) != (i,j) and ((i == m) or (j == l))]
        return constraints

    def AC_3(self):
        """
        Makes the problem arc consistent by applying ac-3 algorithm to arcs (binary constraints).
        :return: nothing but if the problem is unsolvable it will set solution to None
        """
        agenda = []
        # Adding all the arcs
        for variable in self.variables:
            for node in self.constraints[variable]:
                agenda.append((variable, node))
        while len(agenda) != 0:
            (left, right) = agenda.pop()
            if self.revise(left, right):
                if len(self.domains[left]) == 0:
                    self.solution = None
                    return
                for node in self.constraints[left]:
                    if node != right:
                        agenda.append((node, left))
        return True

    def revise(self, left, right):
        revised = False
        for d in self.domains[left].copy():
            if not self.is_consistent(d, right):
                self.domains[left].remove(d)
                revised = True
        return revised

    def is_consistent(self, d, right):
        if self.board[right[0]][right[1]] != 0:
            return d != self.board[right[0]][right[1]]
        return True

    def solve(self):
        assignments = {}
        solution = self.backtrack(assignments)
        if solution is None:
            return None
        self.solved_board = self.solve_board(solution)
        return self.solved_board


    def backtrack(self,assignments):
        if len(assignments) == len(self.variables):
            return assignments
        var = self.MRV(assignments)
        for d in self.domains[var]:
            if self.is_consistent_all(var,d,assignments):
                assignments[var] = d
                solution = self.backtrack(assignments)
                if solution is not None:
                    return solution
                del assignments[var]
        return None

    def is_consistent_all(self,variable,d,assignments):
        for neighbor in self.constraints[variable]:
            if neighbor in assignments and assignments[neighbor] == d:
                return False

        return True

    def MRV(self,assignments):
        """
        :param assignments:
        :return: an unassigned variable with the fewest legal moves
        """
        unassigned_variables = [var for var in self.variables if var not in assignments]
        return min(unassigned_variables,key = lambda  legal_moves : len(self.domains[legal_moves]))


    def solve_board(self,solution):
        n = len(self.board)
        solved = [ [0 for _ in range(n)]for _ in range(n)]
        for tile in self.variables:
            solved[tile[0]][tile[1]] = solution[tile]
        return solved



def solver(board: list[list]) -> list[list]:
    csp = SudokuCSP(board) # define the problem elements
    csp.AC_3()
    return csp.solve()


def main():
    board = [
        [0, 3, 0, 4],
        [0, 1, 0, 0],
        [0, 0, 0, 3],
        [3, 0, 4, 0]
    ]
    print(solver(board))


if __name__ == "__main__":
    main()
