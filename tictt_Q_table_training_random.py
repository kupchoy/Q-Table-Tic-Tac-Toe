"""
This does pretty well and played me to a draw
n_episodes = 50,000
games_won = 40,201

improvements: let it play first for half the time.  for an empty board
it was all still 0.5

play against another agent
"""
# %%
import random
from tqdm import tqdm
from typing import List, Tuple

# %%
winners = [[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [0, 3, 6],
           [1, 4, 7],
           [2, 5, 8],
           [0, 4, 8],
           [2, 4, 6]]


def get_random_cell(b: List[int]) -> int:
    open_cells = []
    for i, c in enumerate(b):
        if c == 0:
            open_cells.append(i)
    return random.choice(open_cells)


def get_block_move(b: List[int]) -> int:
    """
    if there is no block available then block = -1
    :param b:
    :return int:
    """
    block = -1
    choices = [c for c in range(len(b)) if b[c] == 0]
    for c in choices:
        for win_pattern in winners:
            squares = [b[i] for i in win_pattern]
            if (c in win_pattern) and sum(squares) == 2:
                block = c
    return block


def get_win_move(b: List[int]) -> int:
    """
    checks to see if there is a winning move
    if there is no win available then move = -1
    :param b:
    :return int:
    """
    move = -1
    choices = [c for c in range(len(b)) if b[c] == 0]
    for c in choices:
        for win_pattern in winners:
            squares = [b[i] for i in win_pattern]
            if (c in win_pattern) and sum(squares) == -2:
                move = c
    return move


def get_move(b: List[int]) -> int:
    """
    Takes center if open, then check corners, finally random choice
    :param b:
    :return int:
    """
    center = 4
    corners = [0, 2, 6, 8]
    choices = [c for c in range(len(b)) if b[c] == 0]
    if center in choices:
        move = center
        return move
    open_corners = set(choices) & set(corners)
    if len(open_corners) > 0:
        move = random.choice(list(open_corners))
        return move

    return random.choice(choices)


def check_winner(b: List[int]) -> Tuple[bool, int]:
    """
    Tuple of bool, int
    is there a winner: True or False
    who is the winner: 1 for X, -1 for O, 0 for none
    :param b:
    :return:
    """
    for win_pattern in winners:
        square1 = b[win_pattern[0]]
        square2 = b[win_pattern[1]]
        square3 = b[win_pattern[2]]
        if square1 != 0 and square1 == square2 and square2 == square3:
            return True, square1
    return False, 0


def print_board(b: List[str]):
    print(" " + b[0] + " | " + b[1] + " | " + b[2] + "  ")
    print("---|---|---")
    print(" " + b[3] + " | " + b[4] + " | " + b[5] + "  ")
    print("---|---|---")
    print(" " + b[6] + " | " + b[7] + " | " + b[8] + "  ")


# %%


def get_state(b: List[int]) -> int:
    """
    returns the current state, represented as an int
    from 0...|S|-1, where S = set of all possible states
    |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
    some states are not possible, e.g. all cells are x, but we ignore that detail
    this is like finding the integer represented by a base-3 number
    :param b: board
    :return: state hash number
    """
    b2 = [2 if d == -1 else d for d in b]
    return sum([c * 3 ** i for i, c in enumerate(b2)])


def get_state_hash(b: List[int], i=0) -> List:
    results = []

    for v in (0, 1, -1):
        b[i] = v  # if empty board should already be 0
        if i == 8:
            # board is full, collect results and return
            state = get_state(b)
            ended, winner = check_winner(b)
            results.append((state, winner, ended))
        else:
            results += get_state_hash(b, i + 1)

    return results


def get_state_triplet(b: List) -> Tuple:
    state = get_state(b)
    ended, winner = check_winner(b)
    return (state, winner, ended)


def initial_vo(hash: List) -> List:
    """
    initialize state values as follows
    if o wins, V(s) = 1
    if o loses or draw, V(s) = 0
    otherwise, V(s) = 0.5
    :param hash:
    :return: initial Value table
    """
    V = [0] * len(hash)
    for state, winner, ended in hash:
        if ended:
            if winner == -1:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def update_table(history: List, r: int, v: List) -> List:
    """
    we want to BACKTRACK over the states, so that:
    V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
    where V(next_state) = reward if it's the most current state
    :param history:
    :param r:
    :param v:
    :return Updated value table:
    """
    target = r
    for prev in reversed(history):
        value = v[prev] + alpha * (target - v[prev])
        v[prev] = value
        target = value
    return v


def o_agent_move(b: List, v: List) -> int:
    """
    This is only for the O player (-1)
    choose the best action based on current values of states
    loop through all possible moves, get their values
    keep track of the best value
    :param b: board
    :param v: q table
    :return : the best next move
    """
    empty_cells = [c for c in range(len(b)) if b[c] == 0]
    next_move = None
    best_value = -1
    for e in empty_cells:
        # what is the state if we made this move?
        b[e] = -1
        s = get_state(b)
        b[e] = 0  # changing it back
        position_value = v[s]
        if position_value > best_value:
            best_value = position_value
            next_move = e
    return next_move


def o_value_string_board(b: List, sb: List, v: List) -> List:
    """
    This is only for the O player (-1)
    :param b: board
    :param sb: string board
    :param v: q table
    :return : string board with values of empty cells
    """
    empty_cells = [c for c in range(len(b)) if b[c] == 0]
    for e in empty_cells:
        # what is the state if we made this move?
        b[e] = -1
        s = get_state(b)
        b[e] = 0  # changing it back
        position_value = v[s]
        sb[e] = f'{position_value:.2f}'
    return sb


# %%
eps = 0.1  # probability of choosing random action instead of greedy
alpha = 0.5  # learning rate
n_episodes = 50_000
# %%
empty_board = [0 for i in range(0, 9)]
swt = get_state_hash(empty_board)
Vo = initial_vo(swt)
# %%
pbar = tqdm(range(n_episodes))
games_won = 0

for n in pbar:
    board = [0 for i in range(0, 9)]
    numb_moves = 0
    play_game = True
    state_history = []

    while play_game:
        # training so that random goes first same as player would in flutter
        random_cell = get_random_cell(board)
        # update board random choice is -1
        board[random_cell] = 1
        # -------------------------------------
        current_state = get_state(board)
        state_history.append(current_state)
        # -------------------------------------
        is_there_winner, _ = check_winner(board)
        numb_moves += 1
        if is_there_winner or numb_moves == 9:
            if is_there_winner:
                reward = -1
                play_game = False
                break
            else:
                reward = 0
                play_game = False
                break
        # =============== end of human player ============================
        rand_numb = random.random()
        if rand_numb < eps:
            o_move = get_random_cell(board)
            board[o_move] = -1
        else:
            o_move = o_agent_move(board, Vo)
            board[o_move] = -1
        # -------------------------------------
        current_state = get_state(board)
        state_history.append(current_state)
        # -------------------------------------
        is_there_winner, _ = check_winner(board)
        numb_moves += 1
        if is_there_winner:
            reward = 1
            play_game = False
            games_won += 1
    # update table after one game
    Vo = update_table(state_history, reward, Vo)
# %%
board = [0 for i in range(0, 9)]
human_string_board = [str(i) for i in range(9)]
string_board = [' ' for i in range(9)]
numb_moves = 0
play_game = True
state_history = []

while play_game:
    print_board(human_string_board)
    player_choice = int(input('Enter cell'))
    board[player_choice] = 1
    string_board[player_choice] = 'X'
    human_string_board[player_choice] = 'X'
    # -------------------------------------
    current_state = get_state(board)
    state_history.append(current_state)
    # -------------------------------------
    is_there_winner, player = check_winner(board)
    numb_moves += 1
    if is_there_winner or numb_moves == 9:
        if is_there_winner:
            reward = -1
            play_game = False
            print('You Won!')
            print_board(human_string_board)
            break
        else:
            reward = 0
            play_game = False
            print('You Tied')
            print_board(human_string_board)
            break
    # =============== end of human player ============================
    o_string_board = o_value_string_board(board, string_board, Vo)
    print_board(o_string_board)
    o_move = o_agent_move(board, Vo)
    board[o_move] = -1
    string_board[o_move] = 'O'
    human_string_board[o_move] = 'O'
    # -------------------------------------
    current_state = get_state(board)
    state_history.append(current_state)
    # -------------------------------------
    print('agent move: ' + str(o_move))
    is_there_winner, player = check_winner(board)
    numb_moves += 1
    if is_there_winner:
        reward = 1
        play_game = False
        print('You Lost')
        print_board(string_board)
