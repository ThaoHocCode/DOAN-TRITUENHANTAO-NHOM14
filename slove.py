import statistics
import time
import math
import heapq
import random
import copy
from collections import deque
import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class SudokuSolver:
    """
    Lớp cơ sở cho các thuật toán giải Sudoku.
    Chứa các phương thức chung và thuộc tính để theo dõi hiệu suất.
    """
    
    def __init__(self, board, grid_size=9):
        """
        Khởi tạo giải thuật với bảng Sudoku và kích thước lưới.
        
        Args:
            board: Bảng Sudoku 2D (list of lists)
            grid_size: Kích thước lưới (9 hoặc 16)
        """
        self.board = copy.deepcopy(board)
        self.grid_size = grid_size
        self.box_size = 3 if grid_size == 9 else 4
        
        # Các thông số hiệu suất
        self.execution_time = 0
        self.states_explored = 0
        self.max_states_in_memory = 0
        self.h_value = float('inf')  # Giá trị heuristic cuối cùng
        self.g_value = 0  # Số bước thực hiện
        self.f_value = float('inf')  # f(n) = g(n) + h(n)
        
        # Kết quả
        self.solution = None
        self.is_solved = False
    
    def solve(self):
        """
        Phương thức giải Sudoku cần được ghi đè bởi các lớp con.
        """
        raise NotImplementedError("Phương thức này cần được ghi đè bởi lớp con")
    
    def is_valid_move(self, row, col, num):
        """
        Kiểm tra xem việc đặt 'num' tại vị trí (row, col) có hợp lệ không.
        
        Args:
            row: Chỉ số hàng
            col: Chỉ số cột
            num: Giá trị cần kiểm tra
            
        Returns:
            bool: True nếu hợp lệ, False nếu không
        """
        # Kiểm tra hàng
        for j in range(self.grid_size):
            if self.board[row][j] == num:
                return False
        
        # Kiểm tra cột
        for i in range(self.grid_size):
            if self.board[i][col] == num:
                return False
        
        # Kiểm tra hộp
        box_row, box_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if self.board[i][j] == num:
                    return False
        
        return True
    
    def find_empty(self):
        """
        Tìm một ô trống trong bảng.
        
        Returns:
            tuple: (row, col) nếu tìm thấy ô trống, None nếu không có ô trống
        """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def count_empty_cells(self):
        """
        Đếm số ô trống trong bảng.
        
        Returns:
            int: Số ô trống
        """
        count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 0:
                    count += 1
        return count
    
    def calculate_heuristic(self):
        """
        Tính giá trị heuristic h(n) dựa trên số ô trống và số xung đột.
        
        Returns:
            int: Giá trị heuristic
        """
        empty_cells = self.count_empty_cells()
        conflicts = self.count_conflicts()
        return empty_cells + conflicts * 10  # Xung đột được coi là nghiêm trọng hơn
    
    def count_conflicts(self):
        """
        Đếm số xung đột trong bảng (số lần một giá trị xuất hiện nhiều hơn một lần trong hàng/cột/hộp).
        
        Returns:
            int: Số xung đột
        """
        conflicts = 0
        
        # Kiểm tra xung đột trong hàng
        for i in range(self.grid_size):
            values = [0] * (self.grid_size + 1)
            for j in range(self.grid_size):
                if self.board[i][j] != 0:
                    values[self.board[i][j]] += 1
            for val in values:
                if val > 1:
                    conflicts += val - 1
        
        # Kiểm tra xung đột trong cột
        for j in range(self.grid_size):
            values = [0] * (self.grid_size + 1)
            for i in range(self.grid_size):
                if self.board[i][j] != 0:
                    values[self.board[i][j]] += 1
            for val in values:
                if val > 1:
                    conflicts += val - 1
        
        # Kiểm tra xung đột trong hộp
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                values = [0] * (self.grid_size + 1)
                for i in range(box_row, box_row + self.box_size):
                    for j in range(box_col, box_col + self.box_size):
                        if self.board[i][j] != 0:
                            values[self.board[i][j]] += 1
                for val in values:
                    if val > 1:
                        conflicts += val - 1
        
        return conflicts
    
    def get_possible_values(self, row, col):
        """
        Lấy danh sách các giá trị có thể điền vào ô (row, col).
        
        Args:
            row: Chỉ số hàng
            col: Chỉ số cột
            
        Returns:
            list: Danh sách các giá trị hợp lệ
        """
        possible_values = []
        for num in range(1, self.grid_size + 1):
            if self.is_valid_move(row, col, num):
                possible_values.append(num)
        return possible_values
    
    def get_performance_metrics(self):
        """
        Trả về các thông số hiệu suất của thuật toán.
        
        Returns:
            dict: Từ điển chứa các thông số hiệu suất
        """
        return {
            'execution_time': self.execution_time,
            'states_explored': self.states_explored,
            'max_states_in_memory': self.max_states_in_memory,
            'h_value': self.h_value,
            'g_value': self.g_value,
            'f_value': self.f_value,
            'is_solved': self.is_solved
        }


class DFSSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán tìm kiếm theo chiều sâu (DFS).
    """
    
    def solve(self):
        """
        Giải Sudoku bằng DFS.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 0
        
        # Thực hiện DFS
        result = self._dfs()
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn ô trống
        else:
            self.h_value = self.calculate_heuristic()
        
        self.f_value = self.g_value + self.h_value
        
        return result
    
    def _dfs(self):
        """
        Thuật toán DFS đệ quy.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        self.states_explored += 1
        
        # Tìm ô trống
        empty_cell = self.find_empty()
        if not empty_cell:
            return True  # Đã điền hết bảng
        
        row, col = empty_cell
        
        # Tìm các giá trị có thể điền vào ô này (tối ưu hóa)
        possible_values = self.get_possible_values(row, col)
        
        # Thử từng giá trị
        for num in possible_values:
            # Đặt giá trị vào ô
            self.board[row][col] = num
            self.g_value += 1
            
            # Đệ quy để tiếp tục điền các ô khác
            if self._dfs():
                return True
            
            # Nếu không thành công, quay lui
            self.board[row][col] = 0
            self.g_value -= 1
        
        return False


class BFSSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán tìm kiếm theo chiều rộng (BFS).
    """
    
    def solve(self):
        """
        Giải Sudoku bằng BFS.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 0
        
        # Thực hiện BFS
        result = self._bfs()
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn ô trống
        else:
            self.h_value = self.calculate_heuristic()
        
        self.f_value = self.g_value + self.h_value
        
        return result
    
    def _bfs(self):
        """
        Thuật toán BFS sử dụng hàng đợi.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        # Tìm ô trống đầu tiên
        empty_cell = self.find_empty()
        if not empty_cell:
            return True  # Đã điền hết bảng
        
        # Khởi tạo hàng đợi với trạng thái ban đầu
        queue = deque([(copy.deepcopy(self.board), 0)])  # (board, g_value)
        
        # Tập hợp các trạng thái đã xem xét (để tránh trùng lặp)
        visited = set()
        
        while queue:
            self.max_states_in_memory = max(self.max_states_in_memory, len(queue))
            
            current_board, g_value = queue.popleft()
            self.states_explored += 1
            
            # Chuyển bảng thành chuỗi để kiểm tra trùng lặp
            board_str = str(current_board)
            if board_str in visited:
                continue
            
            visited.add(board_str)
            
            # Cập nhật bảng hiện tại
            self.board = copy.deepcopy(current_board)
            self.g_value = g_value
            
            # Tìm ô trống tiếp theo
            empty_cell = self.find_empty()
            if not empty_cell:
                return True  # Đã điền hết bảng
            
            row, col = empty_cell
            
            # Tìm các giá trị có thể điền vào ô này
            possible_values = self.get_possible_values(row, col)
            
            # Thêm các trạng thái mới vào hàng đợi
            for num in possible_values:
                new_board = copy.deepcopy(current_board)
                new_board[row][col] = num
                queue.append((new_board, g_value + 1))
        
        return False


class BacktrackingSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán quay lui (Backtracking) với tối ưu hóa.
    """
    
    def solve(self):
        """
        Giải Sudoku bằng Backtracking.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 1  # Backtracking chỉ lưu một trạng thái tại một thời điểm
        
        # Thực hiện Backtracking
        result = self._backtrack()
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn ô trống
        else:
            self.h_value = self.calculate_heuristic()
        
        self.f_value = self.g_value + self.h_value
        
        return result
    
    def _find_best_empty_cell(self):
        """
        Tìm ô trống với ít giá trị có thể nhất (MRV - Minimum Remaining Values).
        
        Returns:
            tuple: (row, col, possible_values) hoặc None nếu không có ô trống
        """
        min_possibilities = self.grid_size + 1
        best_cell = None
        best_values = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 0:
                    possible_values = self.get_possible_values(i, j)
                    num_possibilities = len(possible_values)
                    
                    if num_possibilities < min_possibilities:
                        min_possibilities = num_possibilities
                        best_cell = (i, j)
                        best_values = possible_values
                        
                        # Nếu chỉ có một giá trị có thể, chọn ngay
                        if num_possibilities == 1:
                            return (i, j, best_values)
        
        if best_cell:
            return (*best_cell, best_values)
        return None
    
    def _backtrack(self):
        """
        Thuật toán Backtracking với tối ưu hóa MRV.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        self.states_explored += 1
        
        # Tìm ô trống tốt nhất (ít giá trị có thể nhất)
        best_empty = self._find_best_empty_cell()
        if not best_empty:
            return True  # Đã điền hết bảng
        
        row, col, possible_values = best_empty
        
        # Nếu không có giá trị nào có thể điền, quay lui
        if not possible_values:
            return False
        
        # Thử từng giá trị
        for num in possible_values:
            # Đặt giá trị vào ô
            self.board[row][col] = num
            self.g_value += 1
            
            # Đệ quy để tiếp tục điền các ô khác
            if self._backtrack():
                return True
            
            # Nếu không thành công, quay lui
            self.board[row][col] = 0
            self.g_value -= 1
        
        return False

class SimulatedAnnealingSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán mô phỏng luyện kim (Simulated Annealing).
    Thuật toán này dựa trên việc tối ưu hóa bằng cách chấp nhận cả những thay đổi làm tăng chi phí
    với một xác suất nhất định, giúp tránh bị mắc kẹt ở cực tiểu cục bộ.
    """
    
    def solve(self):
        """
        Giải Sudoku bằng Simulated Annealing.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 1  # SA chỉ lưu một trạng thái tại một thời điểm
        
        # Lưu lại bảng ban đầu để biết ô nào là cố định
        fixed_cells = self._create_fixed_cells()
        
        # Tạo danh sách các khối (3x3 hoặc 4x4 tùy theo kích thước grid)
        list_of_blocks = self._create_list_of_blocks()
        
        # Khởi tạo bảng với các giá trị hợp lệ trong mỗi khối
        self._randomly_fill_blocks(fixed_cells, list_of_blocks)
        
        # Thực hiện Simulated Annealing
        result = self._simulated_annealing(fixed_cells, list_of_blocks)
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn xung đột
        else:
            self.h_value = self._calculate_number_of_errors()
        
        self.f_value = self.g_value + self.h_value
        
        return result

    def _create_fixed_cells(self):
        """
        Tạo ma trận đánh dấu các ô cố định trong bảng Sudoku.
        Giá trị 1 cho biết ô đó đã có giá trị ban đầu, 0 cho biết ô trống.
        
        Returns:
            list: Ma trận đánh dấu các ô cố định
        """
        fixed_cells = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] != 0:
                    fixed_cells[i][j] = 1
        return fixed_cells
    
    def _create_list_of_blocks(self):
        """
        Tạo danh sách các khối trong bảng Sudoku (3x3 cho lưới 9x9, 4x4 cho lưới 16x16).
        
        Returns:
            list: Danh sách các khối
        """
        final_list_of_blocks = []
        # Xác định kích thước khối dựa trên kích thước lưới
        block_size = self.box_size
        
        for r in range(0, self.grid_size):
            tmp_list = []
            block1 = [i + block_size * ((r) % block_size) for i in range(0, block_size)]
            block2 = [i + block_size * math.trunc((r) / block_size) for i in range(0, block_size)]
            for x in block1:
                for y in block2:
                    tmp_list.append([x, y])
            final_list_of_blocks.append(tmp_list)
        return final_list_of_blocks
    
    def _randomly_fill_blocks(self, fixed_cells, list_of_blocks):
        """
        Điền ngẫu nhiên các giá trị vào các khối của bảng Sudoku.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            list_of_blocks: Danh sách các khối
        """
        for block in list_of_blocks:
            for box in block:
                if self.board[box[0]][box[1]] == 0:  # Nếu ô trống
                    # Lấy các giá trị hiện có trong khối
                    current_block_values = []
                    for b in block:
                        current_block_values.append(self.board[b[0]][b[1]])
                    
                    # Chọn một giá trị ngẫu nhiên không trùng với các giá trị đã có
                    possible_values = [i for i in range(1, self.grid_size + 1) if i not in current_block_values]
                    if possible_values:
                        self.board[box[0]][box[1]] = random.choice(possible_values)
    
    def _calculate_number_of_errors(self):
        """
        Tính tổng số lỗi trên toàn bộ bảng Sudoku (số xung đột trong hàng và cột).
        
        Returns:
            int: Tổng số lỗi
        """
        number_of_errors = 0
        for i in range(self.grid_size):
            number_of_errors += self._calculate_errors_row_column(i, i)
        return number_of_errors
    
    def _calculate_errors_row_column(self, row, column):
        """
        Tính số lỗi (số phần tử trùng lặp) trong một hàng và một cột.
        
        Args:
            row: Chỉ số hàng
            column: Chỉ số cột
            
        Returns:
            int: Số lỗi trong hàng và cột
        """
        # Tính số lỗi trong cột (số giá trị duy nhất thiếu đi từ tổng giá trị)
        col_values = [self.board[i][column] for i in range(self.grid_size) if self.board[i][column] != 0]
        unique_col_values = set(col_values)
        col_errors = len(col_values) - len(unique_col_values)
        
        # Tính số lỗi trong hàng
        row_values = [self.board[row][j] for j in range(self.grid_size) if self.board[row][j] != 0]
        unique_row_values = set(row_values)
        row_errors = len(row_values) - len(unique_row_values)
        
        return col_errors + row_errors
    
    def _two_random_boxes_within_block(self, fixed_cells, block):
        """
        Chọn ngẫu nhiên hai ô không cố định trong một khối.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            block: Danh sách các ô trong một khối
            
        Returns:
            list: Hai ô được chọn hoặc None nếu không tìm thấy
        """
        non_fixed_boxes = [box for box in block if fixed_cells[box[0]][box[1]] != 1]
        
        if len(non_fixed_boxes) < 2:
            return None
            
        first_box = random.choice(non_fixed_boxes)
        second_box = random.choice([box for box in non_fixed_boxes if box != first_box])
        
        return [first_box, second_box]
    
    def _flip_boxes(self, boxes_to_flip):
        """
        Hoán đổi giá trị của hai ô.
        
        Args:
            boxes_to_flip: Hai ô cần hoán đổi
            
        Returns:
            list: Bảng Sudoku sau khi hoán đổi
        """
        # Tạo bản sao của bảng hiện tại
        proposed_board = copy.deepcopy(self.board)
        
        # Hoán đổi giá trị
        place_holder = proposed_board[boxes_to_flip[0][0]][boxes_to_flip[0][1]]
        proposed_board[boxes_to_flip[0][0]][boxes_to_flip[0][1]] = proposed_board[boxes_to_flip[1][0]][boxes_to_flip[1][1]]
        proposed_board[boxes_to_flip[1][0]][boxes_to_flip[1][1]] = place_holder
        
        return proposed_board
    
    def _proposed_state(self, fixed_cells, list_of_blocks):
        """
        Tạo trạng thái đề xuất bằng cách hoán đổi hai ô không cố định trong một khối.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            list_of_blocks: Danh sách các khối
            
        Returns:
            tuple: (bảng mới, các ô đã hoán đổi) hoặc (bảng hiện tại, None) nếu không thể tạo trạng thái mới
        """
        # Xáo trộn danh sách khối để không luôn chọn khối đầu tiên
        shuffled_blocks = random.sample(list_of_blocks, len(list_of_blocks))
        
        for random_block in shuffled_blocks:
            # Tính số ô cố định trong khối
            fixed_count = sum(fixed_cells[box[0]][box[1]] for box in random_block)
            
            # Ngưỡng khác nhau cho lưới 9x9 và 16x16
            max_fixed = 6 if self.grid_size == 9 else 12
            
            # Nếu khối có quá nhiều ô cố định, thử khối khác
            if fixed_count > max_fixed:
                continue
                
            # Chọn hai ô không cố định để hoán đổi
            boxes_to_flip = self._two_random_boxes_within_block(fixed_cells, random_block)
            
            if boxes_to_flip:
                # Tạo bảng mới sau khi hoán đổi
                proposed_board = self._flip_boxes(boxes_to_flip)
                return proposed_board, boxes_to_flip
        
        # Không tìm thấy khối phù hợp
        return self.board, None
    
    def _choose_new_state(self, fixed_cells, list_of_blocks, sigma):
        """
        Quyết định chấp nhận trạng thái mới hay không dựa trên chi phí và nhiệt độ.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            list_of_blocks: Danh sách các khối
            sigma: Nhiệt độ hiện tại
            
        Returns:
            tuple: (bảng mới, thay đổi chi phí)
        """
        # Tạo trạng thái đề xuất
        proposed_board, boxes_to_flip = self._proposed_state(fixed_cells, list_of_blocks)
        
        # Nếu không tìm được trạng thái mới, giữ nguyên trạng thái hiện tại
        if boxes_to_flip is None:
            return self.board, 0
        
        # Lưu trữ bảng hiện tại
        current_board = copy.deepcopy(self.board)
        
        # Tính chi phí hiện tại chỉ cho các hàng và cột liên quan
        current_cost = 0
        for box in boxes_to_flip:
            current_cost += self._calculate_errors_row_column(box[0], box[1])
        
        # Cập nhật bảng tạm thời để tính chi phí mới
        self.board = proposed_board
        
        # Tính chi phí mới chỉ cho các hàng và cột liên quan
        new_cost = 0
        for box in boxes_to_flip:
            new_cost += self._calculate_errors_row_column(box[0], box[1])
        
        # Tính sự thay đổi chi phí
        cost_difference = new_cost - current_cost
        
        # Xác suất chấp nhận theo công thức Metropolis
        rho = math.exp(-cost_difference / sigma)
        
        # Quyết định chấp nhận hay từ chối
        if random.uniform(0, 1) < rho:
            # Chấp nhận trạng thái mới
            return proposed_board, cost_difference
        else:
            # Từ chối, quay lại trạng thái cũ
            self.board = current_board
            return current_board, 0
    
    def _calculate_initial_sigma(self, fixed_cells, list_of_blocks):
        """
        Tính giá trị sigma ban đầu dựa trên độ lệch chuẩn của các chi phí khác nhau.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            list_of_blocks: Danh sách các khối
            
        Returns:
            float: Giá trị sigma ban đầu
        """
        list_of_differences = []
        current_board = copy.deepcopy(self.board)
        
        # Lưu trữ bảng hiện tại
        tmp_board = copy.deepcopy(self.board)
        
        for i in range(10):  # Tạo 10 trạng thái ngẫu nhiên để tính toán
            # Tạo trạng thái đề xuất
            proposed_board, boxes_to_flip = self._proposed_state(fixed_cells, list_of_blocks)
            
            if boxes_to_flip:
                # Cập nhật bảng tạm thời
                self.board = proposed_board
                list_of_differences.append(self._calculate_number_of_errors())
        
        # Khôi phục bảng
        self.board = tmp_board
        
        # Tính độ lệch chuẩn
        if len(list_of_differences) > 1:
            return statistics.pstdev(list_of_differences)
        else:
            return 1.0  # Giá trị mặc định nếu không đủ dữ liệu
    
    def _choose_number_of_iterations(self, fixed_cells):
        """
        Xác định số lần lặp dựa trên số ô cố định.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            
        Returns:
            int: Số lần lặp
        """
        fixed_count = sum(sum(row) for row in fixed_cells)
        # Đảm bảo số lần lặp tăng theo kích thước lưới
        base_iterations = 100
        if self.grid_size == 16:
            base_iterations = 200
            
        return max(fixed_count, base_iterations)
    
    def _simulated_annealing(self, fixed_cells, list_of_blocks):
        """
        Thuật toán Simulated Annealing để giải Sudoku.
        
        Args:
            fixed_cells: Ma trận đánh dấu các ô cố định
            list_of_blocks: Danh sách các khối
            
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        solution_found = False
        attempts = 0
        max_attempts = 3
        
        while not solution_found and attempts < max_attempts:
            attempts += 1
            self.states_explored += 1
            
            # Khôi phục bảng ban đầu và điền ngẫu nhiên ở mỗi lần thử
            if attempts > 1:
                self.board = [[0 if fixed_cells[i][j] == 0 else self.board[i][j] 
                              for j in range(self.grid_size)] 
                              for i in range(self.grid_size)]
                self._randomly_fill_blocks(fixed_cells, list_of_blocks)
            
            # Các tham số cho thuật toán
            decrease_factor = 0.99
            stuck_count = 0
            
            # Tính sigma ban đầu
            sigma = self._calculate_initial_sigma(fixed_cells, list_of_blocks)
            
            # Tính chi phí hiện tại
            score = self._calculate_number_of_errors()
            
            # Số lần lặp
            iterations = self._choose_number_of_iterations(fixed_cells)
            
            # Kiểm tra nếu đã tìm được lời giải
            if score <= 0:
                solution_found = True
                break
            
            # Vòng lặp chính của thuật toán
            max_loops = 500  # Giới hạn số vòng lặp chính
            current_loop = 0
            
            while not solution_found and current_loop < max_loops:
                current_loop += 1
                previous_score = score
                
                for i in range(iterations):
                    # Chọn trạng thái mới
                    new_board, score_diff = self._choose_new_state(fixed_cells, list_of_blocks, sigma)
                    
                    # Cập nhật bảng và chi phí
                    self.board = new_board
                    score += score_diff
                    
                    # Ghi lại trạng thái
                    self.states_explored += 1
                    self.g_value += 1 if score_diff != 0 else 0
                    
                    # Kiểm tra nếu đã tìm được lời giải
                    if score <= 0:
                        solution_found = True
                        break
                
                # Giảm nhiệt độ
                sigma *= decrease_factor
                
                # Kiểm tra nếu đã tìm được lời giải
                if score <= 0:
                    solution_found = True
                    break
                
                # Kiểm tra nếu bị mắc kẹt
                if score >= previous_score:
                    stuck_count += 1
                else:
                    stuck_count = 0
                
                # Tăng nhiệt độ nếu bị mắc kẹt quá lâu
                if stuck_count > 80:
                    sigma += 2
                
                # Kiểm tra lại nếu đã tìm được lời giải
                if self._calculate_number_of_errors() == 0:
                    solution_found = True
                    break
                
                # Tránh vòng lặp vô hạn
                if sigma < 0.001:
                    break
        
        return solution_found

class AStarSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán A*.
    Sử dụng hàm heuristic để tìm đường đi tối ưu nhất đến lời giải.
    """
    
    def solve(self):
        """
        Giải Sudoku bằng A*.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 0
        
        # Thực hiện A*
        result = self._astar()
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn ô trống
        else:
            self.h_value = self.calculate_heuristic()
        
        self.f_value = self.g_value + self.h_value
        
        return result
    
    def _astar(self):
        """
        Thuật toán A* sử dụng hàng đợi ưu tiên.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        # Tìm ô trống đầu tiên
        empty_cell = self.find_empty()
        if not empty_cell:
            return True  # Đã điền hết bảng
        
        # Khởi tạo hàng đợi ưu tiên với trạng thái ban đầu
        # Mỗi phần tử là (f_value, g_value, board)
        priority_queue = [(self.calculate_heuristic(), 0, copy.deepcopy(self.board))]
        heapq.heapify(priority_queue)
        
        # Tập hợp các trạng thái đã xem xét (để tránh trùng lặp)
        visited = set()
        
        while priority_queue:
            self.max_states_in_memory = max(self.max_states_in_memory, len(priority_queue))
            
            # Lấy trạng thái có f_value nhỏ nhất
            f_value, g_value, current_board = heapq.heappop(priority_queue)
            self.states_explored += 1
            
            # Chuyển bảng thành chuỗi để kiểm tra trùng lặp
            board_str = str(current_board)
            if board_str in visited:
                continue
            
            visited.add(board_str)
            
            # Cập nhật bảng hiện tại
            self.board = copy.deepcopy(current_board)
            self.g_value = g_value
            
            # Tìm ô trống tiếp theo
            empty_cell = self.find_empty()
            if not empty_cell:
                return True  # Đã điền hết bảng
            
            row, col = empty_cell
            
            # Tìm các giá trị có thể điền vào ô này
            possible_values = self.get_possible_values(row, col)
            
            # Thêm các trạng thái mới vào hàng đợi ưu tiên
            for num in possible_values:
                new_board = copy.deepcopy(current_board)
                new_board[row][col] = num
                
                # Tính các giá trị heuristic
                self.board = new_board
                h_value = self.calculate_heuristic()
                new_g_value = g_value + 1
                f_value = h_value + new_g_value
                
                # Thêm vào hàng đợi ưu tiên
                heapq.heappush(priority_queue, (f_value, new_g_value, new_board))
        
        return False

class AndOrTreeSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán AND-OR Tree.
    Sử dụng cấu trúc cây AND-OR để biểu diễn và giải quyết các ràng buộc của Sudoku.
    - Nút OR: Các giá trị có thể cho một ô trống
    - Nút AND: Các ràng buộc phải được thỏa mãn đồng thời
    """
    
    def solve(self):
        """
        Giải Sudoku bằng AND-OR Tree.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        start_time = time.time()
        self.states_explored = 0
        self.max_states_in_memory = 0
        self.recursion_depth = 0
        self.max_recursion_depth = 0
        
        # Kiểm tra bảng ban đầu có hợp lệ không
        if not self._is_valid_initial_board():
            self.execution_time = time.time() - start_time
            self.is_solved = False
            return False
        
        # Khởi tạo các cấu trúc dữ liệu để theo dõi các giá trị đã sử dụng
        self.used_in_row = [set() for _ in range(self.grid_size)]
        self.used_in_col = [set() for _ in range(self.grid_size)]
        self.used_in_box = [[set() for _ in range(self.box_size)] for _ in range(self.box_size)]
        
        # Điền các giá trị đã có vào cấu trúc theo dõi
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                num = self.board[i][j]
                if num != 0:
                    self.used_in_row[i].add(num)
                    self.used_in_col[j].add(num)
                    self.used_in_box[i // self.box_size][j // self.box_size].add(num)
        
        # Thực hiện tìm kiếm AND-OR tree
        result = self._and_or_search()
        
        self.execution_time = time.time() - start_time
        self.is_solved = result
        self.solution = copy.deepcopy(self.board) if result else None
        
        # Cập nhật các giá trị heuristic cuối cùng
        if result:
            self.h_value = 0  # Không còn ô trống hoặc xung đột
        else:
            self.h_value = self.calculate_heuristic()
        
        self.f_value = self.g_value + self.h_value
        
        return result
    
    def _is_valid_initial_board(self):
        """
        Kiểm tra bảng ban đầu có hợp lệ không (không có xung đột).
        
        Returns:
            bool: True nếu bảng ban đầu hợp lệ, False nếu không
        """
        # Kiểm tra từng hàng
        for row in range(self.grid_size):
            values = {}
            for col in range(self.grid_size):
                val = self.board[row][col]
                if val != 0:
                    if val in values:
                        return False
                    values[val] = True
        
        # Kiểm tra từng cột
        for col in range(self.grid_size):
            values = {}
            for row in range(self.grid_size):
                val = self.board[row][col]
                if val != 0:
                    if val in values:
                        return False
                    values[val] = True
        
        # Kiểm tra từng hộp
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                values = {}
                for r in range(box_row, box_row + self.box_size):
                    for c in range(box_col, box_col + self.box_size):
                        val = self.board[r][c]
                        if val != 0:
                            if val in values:
                                return False
                            values[val] = True
        
        return True
    
    def _find_best_empty_cell(self):
        """
        Tìm ô trống theo chiến lược MRV (Minimum Remaining Values).
        Tìm ô có ít giá trị hợp lệ nhất để giảm số nhánh cần khám phá.
        
        Returns:
            tuple: (row, col, possible_values) hoặc None nếu không có ô trống
        """
        min_possibilities = self.grid_size + 1
        best_cell = None
        best_values = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 0:
                    # Tìm các giá trị có thể điền vào ô này
                    possible_values = self._get_possible_values_fast(i, j)
                    num_possibilities = len(possible_values)
                    
                    if num_possibilities < min_possibilities:
                        min_possibilities = num_possibilities
                        best_cell = (i, j)
                        best_values = possible_values
                        
                        # Nếu chỉ có một giá trị có thể, chọn ngay
                        if num_possibilities == 1:
                            return (i, j, best_values)
                    
                    # Nếu không còn giá trị hợp lệ cho ô này, trả về ngay
                    if num_possibilities == 0:
                        return (i, j, [])
        
        if best_cell:
            return (*best_cell, best_values)
        return None  # Không còn ô trống
    
    def _get_possible_values_fast(self, row, col):
        """
        Tìm các giá trị có thể điền vào ô (row, col) sử dụng cấu trúc theo dõi đã định nghĩa.
        
        Args:
            row: Chỉ số hàng
            col: Chỉ số cột
            
        Returns:
            list: Danh sách các giá trị hợp lệ
        """
        # Tập hợp các giá trị đã sử dụng trong hàng, cột và hộp
        used_values = self.used_in_row[row].union(
            self.used_in_col[col],
            self.used_in_box[row // self.box_size][col // self.box_size]
        )
        
        # Các giá trị có thể là các giá trị từ 1 đến grid_size mà chưa được sử dụng
        return [num for num in range(1, self.grid_size + 1) if num not in used_values]
    
    def _place_and_check(self, row, col, num):
        """
        Đặt giá trị và kiểm tra các ràng buộc. Thực hiện nút AND trong cây AND-OR.
        
        Args:
            row: Chỉ số hàng
            col: Chỉ số cột
            num: Giá trị cần đặt
            
        Returns:
            bool: True nếu đặt thành công và tất cả ràng buộc đều thỏa mãn, False nếu không
        """
        # Đặt giá trị vào ô
        self.board[row][col] = num
        
        # Cập nhật các cấu trúc theo dõi
        self.used_in_row[row].add(num)
        self.used_in_col[col].add(num)
        self.used_in_box[row // self.box_size][col // self.box_size].add(num)
        
        # Tăng bộ đếm g_value (số bước)
        self.g_value += 1
        
        return True
    
    def _remove_value(self, row, col, num):
        """
        Xóa giá trị khỏi ô và cập nhật các cấu trúc theo dõi. Phục vụ cho quay lui.
        
        Args:
            row: Chỉ số hàng
            col: Chỉ số cột
            num: Giá trị cần xóa
        """
        # Xóa giá trị khỏi các cấu trúc theo dõi
        self.used_in_row[row].remove(num)
        self.used_in_col[col].remove(num)
        self.used_in_box[row // self.box_size][col // self.box_size].remove(num)
        
        # Xóa giá trị khỏi ô
        self.board[row][col] = 0
        
        # Giảm bộ đếm g_value
        self.g_value -= 1
    
    def _and_or_search(self):
        """
        Thuật toán tìm kiếm AND-OR Tree sử dụng MRV và theo dõi các ràng buộc.
        Mỗi nút OR là việc chọn một giá trị từ các giá trị có thể cho một ô.
        Mỗi nút AND là việc đảm bảo tất cả các ràng buộc (hàng, cột, hộp) đều thỏa mãn.
        
        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không
        """
        # Tăng độ sâu đệ quy
        self.recursion_depth += 1
        self.max_recursion_depth = max(self.max_recursion_depth, self.recursion_depth)
        
        # Cập nhật max_states_in_memory
        self.max_states_in_memory = max(self.max_states_in_memory, self.max_recursion_depth)
        
        # Tìm ô trống tốt nhất (MRV)
        best_empty = self._find_best_empty_cell()
        
        # Nếu không còn ô trống, đã tìm thấy lời giải
        if not best_empty:
            self.recursion_depth -= 1
            return True
        
        # Lấy thông tin ô trống và các giá trị có thể
        row, col, possible_values = best_empty
        
        # Nếu không có giá trị nào có thể điền, quay lui
        if not possible_values:
            self.recursion_depth -= 1
            return False
        
        # Duyệt qua các giá trị có thể (nút OR)
        for num in possible_values:
            # Tăng số trạng thái đã khám phá
            self.states_explored += 1
            
            # Đặt giá trị và kiểm tra các ràng buộc (nút AND)
            if self._place_and_check(row, col, num):
                # Tiếp tục đệ quy nếu tất cả ràng buộc đều thỏa mãn
                if self._and_or_search():
                    self.recursion_depth -= 1
                    return True
                
                # Nếu không tìm thấy lời giải, quay lui
                self._remove_value(row, col, num)
        
        # Không tìm thấy lời giải với tất cả các nhánh OR
        self.recursion_depth -= 1
        return False

class SudokuSolver(ABC):
    def __init__(self, board: List[List[int]], grid_size: int):
        self.board = [row[:] for row in board]
        self.grid_size = grid_size
        self.box_size = 3 if grid_size == 9 else 4
        self.states_explored = 0
        self.max_states_in_memory = 0
        self.execution_time = 0
        self.solution = None
        self.h_value = 0
        self.g_value = 0
        self.f_value = 0
        self.is_solved = False

    @abstractmethod
    def solve(self) -> bool:
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "is_solved": self.is_solved,
            "execution_time": self.execution_time,
            "states_explored": self.states_explored,
            "max_states_in_memory": self.max_states_in_memory,
            "h_value": self.h_value,
            "g_value": self.g_value,
            "f_value": self.f_value
        }

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        if self.board[row][col] != 0:
            return False
        for i in range(self.grid_size):
            if self.board[row][i] == num:
                return False
        for i in range(self.grid_size):
            if self.board[i][col] == num:
                return False
        box_row, box_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if self.board[i][j] == num:
                    return False
        return True

    def is_board_valid(self) -> bool:
        for row in range(self.grid_size):
            nums = [self.board[row][col] for col in range(self.grid_size) if self.board[row][col] != 0]
            if len(nums) != len(set(nums)):
                return False
        for col in range(self.grid_size):
            nums = [self.board[row][col] for row in range(self.grid_size) if self.board[row][col] != 0]
            if len(nums) != len(set(nums)):
                return False
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                nums = []
                for row in range(box_row, box_row + self.box_size):
                    for col in range(box_col, box_col + self.box_size):
                        if self.board[row][col] != 0:
                            nums.append(self.board[row][col])
                if len(nums) != len(set(nums)):
                    return False
        return True

class DQNSolver(SudokuSolver):
    """
    Giải Sudoku bằng thuật toán Deep Q-Network (DQN) sử dụng NumPy với constraint propagation.
    """
    
    def __init__(self, board, grid_size=9):
        super().__init__(board, grid_size)
        self.memory = deque(maxlen=5000)  # Tăng kích thước bộ nhớ
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.input_size = grid_size * grid_size
        self.output_size = grid_size * grid_size * grid_size
        self.hidden_size = 128
        self.hidden_size2 = 64
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.max_episodes = 500
        self.max_steps_per_episode = 500
        self.debug = False
        self.original_board = [row[:] for row in board]
        self.possible_values = self._constraint_propagation()
    
    def _build_model(self):
        """
        Xây dựng mạng nơ-ron với hai lớp ẩn.
        """
        W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        b1 = np.zeros((1, self.hidden_size))
        W2 = np.random.randn(self.hidden_size, self.hidden_size2) * np.sqrt(2. / self.hidden_size)
        b2 = np.zeros((1, self.hidden_size2))
        W3 = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2. / self.hidden_size2)
        b3 = np.zeros((1, self.output_size))
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    
    def _forward(self, X, model):
        """
        Lan truyền thuận qua mạng với hai lớp ẩn.
        """
        Z1 = np.dot(X, model['W1']) + model['b1']
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, model['W2']) + model['b2']
        A2 = np.maximum(0, Z2)
        Z3 = np.dot(A2, model['W3']) + model['b3']
        return Z3, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    
    def update_target_model(self):
        """
        Cập nhật mô hình mục tiêu với trọng số của mô hình chính.
        """
        self.target_model = {k: v.copy() for k, v in self.model.items()}
    
    def _constraint_propagation(self):
        """
        Áp dụng constraint propagation để tìm các giá trị có thể cho mỗi ô trống.
        """
        possible_values = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 0:
                    values = [num for num in range(1, self.grid_size + 1) if self.is_valid_move(i, j, num)]
                    possible_values[(i, j)] = values if values else [1]
        return possible_values
    
    def get_empty_cells(self):
        return list(self.possible_values.keys())
    
    def get_valid_moves(self, row, col):
        return self.possible_values.get((row, col), [])
    
    def get_action(self, state: np.ndarray) -> Tuple[int, int, int]:
        empty_cells = self.get_empty_cells()
        if not empty_cells:
            return None, None, None
        
        if np.random.rand() <= self.epsilon:
            row, col = random.choice(empty_cells)
            valid_nums = self.get_valid_moves(row, col)
            num = random.choice(valid_nums) if valid_nums else random.randint(1, self.grid_size)
            return row, col, num
        else:
            state_flat = state.reshape(1, -1)
            q_values, _ = self._forward(state_flat, self.model)
            q_values = q_values[0]
            
            best_action = None
            best_q = -float('inf')
            empty_cells_with_constraints = [(row, col, len(self.get_valid_moves(row, col))) for row, col in empty_cells]
            empty_cells_with_constraints.sort(key=lambda x: x[2])
            
            for row, col, _ in empty_cells_with_constraints:
                valid_nums = self.get_valid_moves(row, col)
                if not valid_nums:
                    continue
                for num in valid_nums:
                    action_idx = (row * self.grid_size + col) * self.grid_size + (num - 1)
                    if action_idx < len(q_values) and q_values[action_idx] > best_q:
                        best_q = q_values[action_idx]
                        best_action = (row, col, num)
            
            if best_action is None and empty_cells:
                row, col = random.choice(empty_cells)
                valid_nums = self.get_valid_moves(row, col)
                num = random.choice(valid_nums) if valid_nums else random.randint(1, self.grid_size)
                best_action = (row, col, num)
            
            return best_action if best_action else (None, None, None)
    
    def store_transition(self, state: np.ndarray, action: Tuple[int, int, int], reward: float, next_state: np.ndarray, done: bool):
        if action[0] is not None:
            self.memory.append((state, action, reward, next_state, done))
            self.max_states_in_memory = max(self.max_states_in_memory, len(self.memory))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        X = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        
        current_q_values, cache = self._forward(X, self.model)
        next_q_values, _ = self._forward(next_states, self.target_model)
        
        targets = current_q_values.copy()
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            row, col, num = action
            action_idx = (row * self.grid_size + col) * self.grid_size + (num - 1)
            targets[i, action_idx] = reward if done else reward + self.gamma * np.max(next_q_values[i])
        
        dZ3 = current_q_values - targets
        dW3 = np.dot(cache['A2'].T, dZ3) / self.batch_size
        db3 = np.sum(dZ3, axis=0, keepdims=True) / self.batch_size
        dA2 = np.dot(dZ3, self.model['W3'].T)
        dZ2 = dA2 * (cache['Z2'] > 0)
        dW2 = np.dot(cache['A1'].T, dZ2) / self.batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / self.batch_size
        dA1 = np.dot(dZ2, self.model['W2'].T)
        dZ1 = dA1 * (cache['Z1'] > 0)
        dW1 = np.dot(X.T, dZ1) / self.batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / self.batch_size
        
        clip_value = 10.0
        for grad in [dW1, dW2, dW3, db1, db2, db3]:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        self.model['W1'] -= self.learning_rate * dW1
        self.model['b1'] -= self.learning_rate * db1
        self.model['W2'] -= self.learning_rate * dW2
        self.model['b2'] -= self.learning_rate * db2
        self.model['W3'] -= self.learning_rate * dW3
        self.model['b3'] -= self.learning_rate * db3
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_state_representation(self, board=None):
        if board is None:
            board = self.board
        state = np.array(board).flatten()
        state = state / self.grid_size
        return state
    
    def evaluate_board(self):
        conflicts = 0
        for i in range(self.grid_size):
            row_nums = [self.board[i][j] for j in range(self.grid_size) if self.board[i][j] != 0]
            conflicts += len(row_nums) - len(set(row_nums))
            col_nums = [self.board[j][i] for j in range(self.grid_size) if self.board[j][i] != 0]
            conflicts += len(col_nums) - len(set(col_nums))
        
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                nums = [self.board[i][j] for i in range(box_row, box_row + self.box_size)
                        for j in range(box_col, box_col + self.box_size) if self.board[i][j] != 0]
                conflicts += len(nums) - len(set(nums))
        
        filled_cells = sum(1 for row in self.board for cell in row if cell != 0)
        fill_rate = filled_cells / (self.grid_size * self.grid_size)
        conflict_score = 1.0 / (1.0 + conflicts)
        
        return 0.6 * conflict_score + 0.4 * fill_rate
    
    def solve(self) -> bool:
        start_time = time.time()
        self.solution = [row[:] for row in self.board]
        best_board = None
        best_score = 0
        original_board = [row[:] for row in self.board]
        
        for episode in range(self.max_episodes):
            self.board = [row[:] for row in original_board]
            self.possible_values = self._constraint_propagation()
            state = self.get_state_representation()
            steps = 0
            
            while steps < self.max_steps_per_episode:
                self.states_explored += 1
                action = self.get_action(state)
                
                if action[0] is None:
                    break
                
                row, col, num = action
                valid_move = self.is_valid_move(row, col, num)
                if valid_move:
                    old_board = [row[:] for row in self.board]
                    self.board[row][col] = num
                    board_valid = self.is_board_valid()
                    
                    if not board_valid:
                        self.board = old_board
                        reward = -5
                    else:
                        empty_cells = self.get_empty_cells()
                        current_score = self.evaluate_board()
                        reward = 1 + current_score
                        
                        if not empty_cells:
                            reward = 100
                            done = True
                            is_valid_solution = self.is_board_valid()
                            if is_valid_solution:
                                self.solution = [row[:] for row in self.board]
                                best_board = self.solution
                                best_score = 1.0
                                if self.debug:
                                    print(f"Đã tìm thấy lời giải ở episode {episode + 1}, bước {steps + 1}")
                                break
                        else:
                            if current_score > best_score:
                                best_score = current_score
                                best_board = [row[:] for row in self.board]
                else:
                    reward = -1
                
                next_state = self.get_state_representation()
                done = len(self.get_empty_cells()) == 0
                self.store_transition(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                steps += 1
                
                if done:
                    break
            
            if episode % 5 == 0:
                self.update_target_model()
            
            if best_score == 1.0:
                break
        
        self.execution_time = time.time() - start_time
        if best_board:
            self.solution = best_board
        
        self.is_solved = (best_score == 1.0) or (
            self.solution is not None and 
            all(self.solution[i][j] != 0 for i in range(self.grid_size) for j in range(self.grid_size)) and
            self.is_board_valid()
        )
        
        if self.debug:
            print(f"Đã giải {'thành công' if self.is_solved else 'thất bại'} sau {self.execution_time:.2f} giây, điểm: {best_score:.4f}")
        
        return self.is_solved
        
def get_solver(algorithm, board, grid_size=9):
    """
    Trả về đối tượng giải thuật tương ứng với thuật toán được chọn.
    
    Args:
        algorithm: Tên thuật toán ('dfs', 'bfs', 'backtracking', 'simulated_annealing', 'A*', 'and_or')
        board: Bảng Sudoku 2D
        grid_size: Kích thước lưới (9 hoặc 16)
        
    Returns:
        SudokuSolver: Đối tượng giải thuật
    """
    if algorithm == 'dfs':
        return DFSSolver(board, grid_size)
    elif algorithm == 'bfs':
        return BFSSolver(board, grid_size)
    elif algorithm == 'backtracking':
        return BacktrackingSolver(board, grid_size)
    elif algorithm == 'simulated_annealing':
        return SimulatedAnnealingSolver(board, grid_size)
    elif algorithm == 'A*':
        return AStarSolver(board, grid_size)
    elif algorithm == 'and_or':
        return AndOrTreeSolver(board, grid_size)
    elif algorithm == 'dqn':
        return DQNSolver(board, grid_size)
    else:
        raise ValueError(f"Thuật toán không hợp lệ: {algorithm}")


