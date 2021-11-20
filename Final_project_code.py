from __future__ import print_function
import re, sys, time
from itertools import count
from collections import namedtuple
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import time
import os
import glob
from tensorflow.keras.models import load_model

model = load_model('/home/tejas/Pictures/model_50x50.hd5')
label_map = list('_kqrbnp')
arr = [['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
       ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
       ['_', '_', '_', '_', '_', '_', '_', '_'],
       ['_', '_', '_', '_', '_', '_', '_', '_'],
       ['_', '_', '_', '_', '_', '_', '_', '_'],
       ['_', '_', '_', '_', '_', '_', '_', '_'],
       ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
       ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']]

# After array
mat1 = [['1', '1', '1', '1', '1', '1', '1', '1'],
        ['1', '1', '1', '1', '1', '1', '1', '1'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['2', '2', '2', '2', '2', '2', '2', '2'],
        ['2', '2', '2', '2', '2', '2', '2', '2']]

# Before Array
mat2 = [['1', '1', '1', '1', '1', '1', '1', '1'],
        ['1', '1', '1', '1', '1', '1', '1', '1'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['2', '2', '2', '2', '2', '2', '2', '2'],
        ['2', '2', '2', '2', '2', '2', '2', '2']]


# !/usr/bin/env pypy
# -*- coding: utf-8 -*-

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################


def predict(img, model, img_size=(50, 50), plot=False):
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # if plot:
    #   plt.imshow(img, cmap='gray')

    img = img.reshape(1, *img_size, 1) / 255
    pred = model.predict(img)
    return label_map[np.argmax(pred)]

name_map = {
    '_': 'Empty_Square',
    'k': 'Black_King',
    'q': 'Black_Queen',
    'r': 'Black_Rook',
    'b': 'Black_Bishop',
    'n': 'Black_Knight',
    'p': 'Black_Pawn',
}


def get_square(img, row, col):
    width = img.shape[0]
    print(width)
    square = width // 8
    print(square)
    x1, y1 = row * square, col * square
    print(x1, y1)
    x2, y2 = x1 + square, y1 + square
    print(x2, y2)
    return img[x1:x2, y1:y2]


def correct_perspective(img, approx):
    pts = approx.reshape(4, 2)
    tl, tr, br, bl = (
        pts[np.argmin(np.sum(pts, axis=1))],
        pts[np.argmin(np.diff(pts, axis=1))],
        pts[np.argmax(np.sum(pts, axis=1))],
        pts[np.argmax(np.diff(pts, axis=1))]
    )

    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    src = np.array([tl, tr, br, bl], dtype='float32')
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                   dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (int(w), int(h)))

    return cv2.resize(img, (400, 400))


def find_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # superfluous dimensions
    children = []

    for i in range(len(contours)):
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            children.append(hierarchy[i])

    children = np.array(children)
    values, counts = np.unique(children[:, 3], return_counts=True)
    contour = contours[values[np.argmax(counts)]]
    epsilon = 0.1 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_frames():
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    if not capture.isOpened():
        raise RuntimeError('Failed to start camera.')

    while True:
        ret, img = capture.read()
        cv2.imshow('webcam', img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    capture.release()

    return img


piece = {'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000}
pst = {
    'P': (0, 0, 0, 0, 0, 0, 0, 0,
          78, 83, 86, 73, 102, 82, 85, 90,
          7, 29, 21, 44, 40, 31, 44, 7,
          -17, 16, -2, 15, 14, 0, 15, -13,
          -26, 3, 10, 9, 6, 1, 0, -23,
          -22, 9, 5, -11, -10, -2, 3, -19,
          -31, 8, -7, -37, -36, -14, 3, -31,
          0, 0, 0, 0, 0, 0, 0, 0),
    'N': (-66, -53, -75, -75, -10, -55, -58, -70,
          -3, -6, 100, -36, 4, 62, -4, -14,
          10, 67, 1, 74, 73, 27, 62, -2,
          24, 24, 45, 37, 33, 41, 25, 17,
          -1, 5, 31, 21, 22, 35, 2, 0,
          -18, 10, 13, 22, 18, 15, 11, -14,
          -23, -15, 2, 0, 2, 0, -23, -20,
          -74, -23, -26, -24, -19, -35, -22, -69),
    'B': (-59, -78, -82, -76, -23, -107, -37, -50,
          -11, 20, 35, -42, -39, 31, 2, -22,
          -9, 39, -32, 41, 52, -10, 28, -14,
          25, 17, 20, 34, 26, 25, 15, 10,
          13, 10, 17, 23, 17, 16, 0, 7,
          14, 25, 24, 15, 8, 25, 20, 15,
          19, 20, 11, 6, 7, 6, 20, 16,
          -7, 2, -15, -12, -14, -15, -10, -10),
    'R': (35, 29, 33, 4, 37, 33, 56, 50,
          55, 29, 56, 67, 55, 62, 34, 60,
          19, 35, 28, 33, 45, 27, 25, 15,
          0, 5, 16, 13, 18, -4, -9, -6,
          -28, -35, -16, -21, -13, -29, -46, -30,
          -42, -28, -42, -25, -25, -35, -26, -46,
          -53, -38, -31, -26, -29, -43, -44, -53,
          -30, -24, -18, 5, -2, -18, -31, -32),
    'Q': (6, 1, -8, -104, 69, 24, 88, 26,
          14, 32, 60, -10, 20, 76, 57, 24,
          -2, 43, 32, 60, 72, 63, 43, 2,
          1, -16, 22, 17, 25, 20, -13, -6,
          -14, -15, -2, -5, -1, -10, -20, -22,
          -30, -6, -13, -11, -16, -11, -16, -27,
          -36, -18, 0, -19, -15, -15, -21, -38,
          -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (4, 54, 47, -99, -99, 60, 83, -62,
          -32, 10, 55, 56, 56, 55, 10, 3,
          -62, 12, -57, 44, -67, 28, 37, -31,
          -55, 50, 11, -4, -19, 13, 0, -49,
          -55, -43, -52, -28, -51, -47, -8, -50,
          -47, -42, -43, -79, -64, -32, -29, -32,
          -4, 3, -14, -50, -57, -18, 13, 4,
          17, 30, -3, -14, 6, -1, 40, 18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8:i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  # 0 -  9
    '         \n'  # 10 - 19
    ' rnbqkbnr\n'  # 20 - 29
    ' pppppppp\n'  # 30 - 39
    ' ........\n'  # 40 - 49
    ' ........\n'  # 50 - 59
    ' ........\n'  # 60 - 69
    ' ........\n'  # 70 - 79
    ' PPPPPPPP\n'  # 80 - 89
    ' RNBQKBNR\n'  # 90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N + N, N + W, N + E),
    'N': (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    'B': (N + E, S + E, S + W, N + W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N + E, S + E, S + W, N + W),
    'K': (N, E, S, W, N + E, S + E, S + W, N + W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10 * piece['Q']
MATE_UPPER = piece['K'] + 10 * piece['Q']

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N + N) and q != '.': break
                    if p == 'P' and d == N + N and (i < A1 + N or self.board[i + N] != '.'): break
                    if p == 'P' and d in (N + W, N + E) and q == '.' \
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == 'K' and self.wc[0]: yield (j + E, j + W)
                    if i == H1 and self.board[j + W] == 'K' and self.wc[1]: yield (j + W, j + E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep else 0,
            119 - self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += pst['K'][119 - j]
        # Castling
        if p == 'K' and abs(i - j) == 2:
            score += pst['R'][(i + j) // 2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119 - (j + S)]
        return score


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # FIXME: This is not true, since other positions will be affected by
        # the new values for all the drawn positions.
        if DRAW_TEST:
            if not root and pos in self.history:
                return 0

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major piece left on the board, since otherwise zugzwangs are too dangerous.
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1 - gamma, depth - 3, root=False)
            # For QSearch we have a different kind of null-move, namely we can just stop and not capture anythign else.
            if depth == 0:
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us. Note, we don't have to check for legality, since we've already done it before. Also note that in QS the killer must be a capture, otherwise we will be non deterministic.
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1, root=False)
            # Then all the other moves
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                # If depth == 0 we only try moves with high intrinsic score (captures and promotions). Otherwise we do all moves.
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Clear before setting, so we always have a value
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.
        # This doesn't prevent sunfish from making a move that results in stalemate,
        # but only if depth == 1, so that's probably fair enough.
        # (Btw, at depth 1 we can also mate without realizing.)
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)

        return best

    def search(self, pos, history=()):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0
        if DRAW_TEST:
            self.history = set(history)
            # print('# Clearing table due to new history')
            self.tp_score.clear()

        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower + upper + 1) // 2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            # We want to make sure the move to play hasn't been kicked out of the table,
            # So we make another call that must always fail high and thus produce a move.
            self.bound(pos, lower, depth)
            # If the game hasn't finished we can retrieve our move from the
            # transposition table.
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
                  'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙', '.': '·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8 - i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


def main():
    hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
    searcher = Searcher()
    while True:
        img = get_frames()
        plt.figure(figsize=(10, 10))
        plt.xlabel('Webcam Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

        plt.figure(figsize=(15, 15))

        plt.subplot(2, 2, 1)
        plt.imshow(thresh, cmap='gray')
        plt.xlabel('Threshold')

        plt.subplot(2, 2, 2)
        plt.imshow(contour_img)
        plt.xlabel('Contours')

        plt.tight_layout()
        plt.show()

        approx = find_board_contour(img)
        contour_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 3)

        plt.figure(figsize=(10, 10))
        plt.xlabel('Board Contour')
        plt.imshow(contour_img)
        plt.show()

        plt.figure(figsize=(8, 8))

        board = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        board = correct_perspective(board, approx)
        plt.imshow(board)
        plt.show()

        # fig = plt.figure(1, (8,8))
        # grid = ImageGrid(fig, 111, nrows_ncols=(8,8), axes_pad=0.1)

        board = correct_perspective(img, approx)
        board = cv2.resize(board, (400, 400))
        for i in range(8 * 8):
            col = i % 8
            row = i // 8
            square = get_square(board, row, col)
            # plt.imshow(square)
            # plt.show()
            # file_name = '{}_{}.jpg'.format(square, time.time())
            # cv2.imwrite(os.path.join(path, file_name), square)
            # grid[i].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
            a = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            pred = predict(a, model, plot=True)
            print(name_map[pred])
            #print(pred)
            # print(pred)
            if (name_map[pred] == 'Black_King' or name_map[pred] == 'Black_Queen' or name_map[pred] == 'Black_Rook' or name_map[pred] == 'Black_Bishop' or name_map[pred] == 'Black_Knight' or name_map[pred] == 'Black_Pawn'):
                mat1[row][col] = '2'
        for i in range(8 * 8):
            col=i % 8
            row=i // 8
            if(mat1[row][col]=='2'):
                square=get_square(board,row,col)
                a=cv2.cvtColor(square,cv2.COLOR_BGR2RGB)
                pred=predict(a,model,plot=True)
                if(name_map[pred]=='Empty_Square'):
                    mat1[row][col]='_'
        b1 = 0
        b2 = 0
        b3 = 0
        b4 = 0
        change = [[0, 0], [0, 0]]
        flag = 0
        for i in range(0, 8):
            for j in range(0, 8):
                if (mat1[i][j] != mat2[i][j]):
                    change[flag][0] = i
                    change[flag][1] = j
                    # print(change)       ######################
                    flag += 1
                    if (flag > 1):
                        #print("Invalid move!")
                        flag = 0
                        break

        if (mat2[change[0][0]][change[0][1]] == "_"):
            change[1], change[0] = change[0], change[1]
        # print("Change")
        # print(change)
        # print("Before ")
        #for i in range(len(arr)):
         #   for j in range(len(arr[i])):
         #       print(arr[i][j], end=" ")
         #   print()  ################################################################################################

        # if(mat1[change[1][0]][change[1][1] != "_"]):
        #   mat2[change[1][0]][change[1][1]] = "_"
        
        arr[change[0][0]][change[0][1]], arr[change[1][0]][change[1][1]] = arr[change[1][0]][change[1][1]],arr[change[0][0]][change[0][1]]
        mat2[change[0][0]][change[0][1]], mat2[change[1][0]][change[1][1]] = mat2[change[1][0]][change[1][1]],mat2[change[0][0]][change[0][1]]

       
        #mat1[change[0][0]][change[0][1]]='_'
        print("Matrix1")
        for i in range(len(mat1)):
            for j in range(len(mat1[i])):
                print(mat1[i][j], end=" ")
            print()

        print("Matrix2")
        for i in range(len(mat2)):
            for j in range(len(mat2[i])):
                print(mat2[i][j], end=" ")
            print()
       # print(change[1][0],change[1][1],change[0][0],change[0][1])
        print(arr[change[1][0]][change[1][1]], "is moved from", change[0], " to ", change[1])
       # print(change[1][0],change[1][1])
       # print(change[0][0],change[0][1])
        
        if (change[1][0] == 0):
            a1 = str(8)
        elif (change[1][0] == 1):
            a1 = str(7)
        elif (change[1][0] == 2):
            a1 = str(6)
        elif (change[1][0] == 3):
            a1 = str(5)
        elif (change[1][0] == 4):
            a1 = str(4)
        elif (change[1][0] == 5):
            a1 = str(3)
        elif (change[1][0] == 6):
            a1 = str(2)
        elif (change[1][0] == 7):
            a1 = str(1)

        if (change[1][1] == 0):
            a2 = 'a'
        elif (change[1][1] == 1):
            a2 = 'b'
        elif (change[1][1] == 2):
            a2 = 'c'
        elif (change[1][1] == 3):
            a2 = 'd'
        elif (change[1][1] == 4):
            a2 = 'e'
        elif (change[1][1] == 5):
            a2 = 'f'
        elif (change[1][1] == 6):
            a2 = 'g'
        elif (change[1][1] == 7):
            a2 = 'h'

        if (change[0][0] == 0):
            a3 = str(8)
        elif (change[0][0] == 1):
            a3 = str(7)
        elif (change[0][0] == 2):
            a3 = str(6)
        elif (change[0][0] == 3):
            a3 = str(5)
        elif (change[0][0] == 4):
            a3 = str(4)
        elif (change[0][0] == 5):
            a3 = str(3)
        elif (change[0][0] == 6):
            a3 = str(2)
        elif (change[0][0] == 7):
            a3 = str(1)

        if (change[0][1] == 0):
            a4 = 'a'
        elif (change[0][1] == 1):
            a4 = 'b'
        elif (change[0][1] == 2):
            a4 = 'c'
        elif (change[0][1] == 3):
            a4 = 'd'
        elif (change[0][1] == 4):
            a4 = 'e'
        elif (change[0][1] == 5):
            a4 = 'f'
        elif (change[0][1] == 6):
            a4 = 'g'
        elif (change[0][1] == 7):
            a4 = 'h'


        print("Your move:")  
        print(a2 + a1 + a4 + a3)
        
        A = a4 + a3 + a2 + a1


        print_pos(hist[-1])

        if hist[-1].score <= -MATE_LOWER:
            print("You lost")
            break

# We query the user until she enters a (pseudo) legal move.
        move = None
        while move not in hist[-1].gen_moves():
            match = re.match('([a-h][1-8])' * 2, A)
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
        # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        hist.append(hist[-1].move(move))

# After our move we rotate the board and print it again.
# This allows us to see the effect of our move.
        print_pos(hist[-1].rotate())

        if hist[-1].score <= -MATE_LOWER:
            print("You won")
            break

# Fire up the engine to look for a move.
        start = time.time()
        for _depth, move, score in searcher.search(hist[-1], hist):
            if time.time() - start > 1:
                break

        if score == MATE_UPPER:
            print("Checkmate!")

# The black player moves from a rotated position, so we have to
# 'back rotate' the move before printing it.
        print("AI's move:", render(119 - move[0]) + render(119 - move[1]))
        u = render(119 - move[0])
        print(u[0], u[1])
        v = render(119 - move[1])
        print(v[0], v[1])
        hist.append(hist[-1].move(move))
        if (u[0] == 'a'):
            b1 = 0
        elif (u[0] == 'b'):
            b1 = 1
        elif (u[0] == 'c'):
            b1 = 2
        elif (u[0] == 'd'):
            b1 = 3
        elif (u[0] == 'e'):
            b1 = 4
        elif (u[0] == 'f'):
            b1 = 5
        elif (u[0] == 'g'):
            b1 = 6
        elif (u[0] == 'h'):
            b1 = 7
        
        if (u[1] == str(1)):
            b2 = 7
        elif (u[1] == str(2)):
            b2 = 6
        elif (u[1] == str(3)):
            b2 = 5
        elif (u[1] == str(4)):
            b2 = 4
        elif (u[1] == str(5)):
            b2 = 3
        elif (u[1] == str(6)):
            b2 = 2
        elif (u[1] == str(7)):
            b2 = 1
        elif (u[1] == str(8)):
            b2 = 0
        
        if (v[0] == 'a'):
            b3 = 0
        elif (v[0] == 'b'):
            b3 = 1
        elif (v[0] == 'c'):
            b3 = 2
        elif (v[0] == 'd'):
            b3 = 3
        elif (v[0] == 'e'):
            b3 = 4
        elif (v[0] == 'f'):
            b3 = 5
        elif (v[0] == 'g'):
            b3 = 6
        elif (v[0] == 'h'):
            b3 = 7

        if (v[1] == str(1)):
            b4 = 7
        elif (v[1] == str(2)):
            b4 = 6
        elif (v[1] == str(3)):
            b4 = 5
        elif (v[1] == str(4)):
            b4 = 4
        elif (v[1] == str(5)):
            b4 = 3
        elif (v[1] == str(6)):
            b4 = 2
        elif (v[1] == str(7)):
            b4 = 1
        elif (v[1] == str(8)):
            b4 = 0
        print(b2, b1, b4, b3)
        mat1[b2][b1] = '_'
        mat1[b4][b3] = 1
        mat2[b2][b1] = '_'
        mat2[b4][b3] = 1
        print("*****Matrix1*****")
        for i in range(len(mat1)):
            for j in range(len(mat1[i])):
                print(mat1[i][j], end=" ")
            print()

        print("*****Matrix2*****")
        for i in range(len(mat2)):
            for j in range(len(mat2[i])):
                print(mat2[i][j], end=" ")
            print()
if __name__ == '__main__':
    main()

