import tkinter as tk
from tkinter import ttk, messagebox, StringVar, Frame, Label, Button, Entry, Toplevel
import ttkbootstrap as ttkb  
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SelectionScreen:
    """Màn hình chọn kích thước lưới và độ khó"""
    
    def __init__(self, master, start_callback):
        self.master = master
        self.start_callback = start_callback
        
        
        self.style = ttkb.Style(theme="flatly")
        
        self.master.configure(bg=self.style.colors.bg)
        
        container = ttkb.Frame(self.master, padding=20)
        container.pack(expand=True)
        
        title_label = ttkb.Label(container, text="SUDOKU", font=("Arial", 36, "bold"),
                                bootstyle="primary")
        title_label.pack(pady=(0, 30))
        
        options_frame = ttkb.LabelFrame(container, text="Tùy chọn trò chơi", padding=15,
                                      bootstyle="info")
        options_frame.pack(fill="x", pady=10)
        
        grid_frame = ttkb.Frame(options_frame)
        grid_frame.pack(fill="x", pady=10)
        
        ttkb.Label(grid_frame, text="Kích thước lưới:", font=("Arial", 14)).pack(side="left")
        self.grid_size_var = tk.StringVar(value="9x9")  
        grid_menu = ttkb.Combobox(grid_frame, textvariable=self.grid_size_var,
                                 values=["9x9", "16x16"], state="readonly",
                                 bootstyle="primary", width=10)
        grid_menu.pack(side="left", padx=10)
        
        difficulty_frame = ttkb.Frame(options_frame)
        difficulty_frame.pack(fill="x", pady=10)
        
        ttkb.Label(difficulty_frame, text="Độ khó:", font=("Arial", 14)).pack(side="left")
        self.difficulty_var = tk.StringVar(value="Medium")  
        self.difficulty_map = {
            "Super Easy": "super_easy", "Easy": "easy", "Medium": "medium",
            "Difficult": "difficult", "Expert": "expert"
        }
        difficulty_menu = ttkb.Combobox(difficulty_frame, textvariable=self.difficulty_var,
                                       values=list(self.difficulty_map.keys()), state="readonly",
                                       bootstyle="primary", width=12)
        difficulty_menu.pack(side="left", padx=10)
        
        start_button = ttkb.Button(container, text="Bắt Đầu Chơi", command=self._start_game,
                                 bootstyle="success-outline", padding=(20, 10))
        start_button.pack(pady=30)
        
        instructions = """
        Hướng dẫn:
        - Chọn kích thước lưới (9x9 hoặc 16x16)
        - Chọn độ khó phù hợp
        - Nhấn "Bắt Đầu Chơi" để bắt đầu
        """
        ttkb.Label(container, text=instructions, font=("Arial", 10),
                  bootstyle="secondary").pack(pady=10, anchor="w")
    
    def _start_game(self):
        grid_size_text = self.grid_size_var.get()
        grid_size = 9 if grid_size_text == "9x9" else 16
        difficulty_display = self.difficulty_var.get()
        difficulty = self.difficulty_map[difficulty_display]
        self.start_callback(grid_size, difficulty, difficulty_display)

class GameScreen:
    """Giao diện chơi game Sudoku"""
    
    def __init__(self, master, grid_size=9):
        self.master = master
        self.master.title("Sudoku Game")
        
        # Màu sắc
        self.bg_color = "#F0F0F0"
        self.fg_color = "#333333"
        self.highlight_color = "#FFEB99"  
        self.error_color = "#FF8888"     
        self.success_color = "#88FF88"    
        self.original_cell_color = "#DDDDDD" 
        
        self.master.configure(bg=self.bg_color)
        
        
        self.cells = []  
        self.cell_vars = []  
        self.original_cells = set()  
        self.timer_var = StringVar(value="Thời gian: 00:00")
        self.lives_var = StringVar(value="Mạng: 3")
        self.timer_running = False
        
        self.grid_size = grid_size
        
        # Tạo giao diện
        self._create_header()
        self._create_info_panel()
        self._create_board()
        self._create_controls()
        self._create_status_bar()
        
        # Bắt đầu bộ đếm thời gian
        self._update_timer()
    
    def _create_header(self):
        """Tạo phần header với tiêu đề và nút quay lại"""
        header_frame = Frame(self.master, bg=self.bg_color)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Nút quay lại
        self.back_button = Button(header_frame, text="Quay lại", font=("Arial", 10),
                                 command=self._on_back, bg="#E74C3C", fg="white")
        self.back_button.pack(side="left", padx=5)
        
        # Tiêu đề
        Label(header_frame, text="SUDOKU", font=("Arial", 18, "bold"),
             bg=self.bg_color, fg=self.fg_color).pack(side="top", pady=5)
    
    def _create_info_panel(self):
        """Tạo panel hiển thị thời gian và mạng"""
        info_frame = Frame(self.master, bg=self.bg_color)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Hiển thị thời gian
        timer_label = Label(info_frame, textvariable=self.timer_var, 
                          font=("Arial", 12, "bold"), bg=self.bg_color, fg="#2980B9")
        timer_label.pack(side="left", padx=20)
        
        # Hiển thị mạng
        lives_label = Label(info_frame, textvariable=self.lives_var, 
                          font=("Arial", 12, "bold"), bg=self.bg_color, fg="#E74C3C")
        lives_label.pack(side="right", padx=20)
    
    def _create_board(self):
        """Tạo bảng Sudoku (9x9 hoặc 16x16) với giao diện dễ nhìn hơn cho 16x16"""
        # Nếu board_frame đã tồn tại, xóa nó
        if hasattr(self, 'board_frame'):
            self.board_frame.destroy()
    
        self.board_frame = Frame(self.master, bg="black", padx=2, pady=2)
        self.board_frame.pack(padx=10, pady=10)
    
        # Tạo lưới ô
        self.cells = []
        self.cell_vars = []
    
        # Xác định kích thước ô dựa trên kích thước lưới
        box_size = 3 if self.grid_size == 9 else 4
    
        # Điều chỉnh kích thước ô và cỡ chữ
        cell_font_size = 18 if self.grid_size == 9 else 14  # Tăng cỡ chữ cho 16x16
        cell_width = 2 if self.grid_size == 9 else 1        # Giữ ô hẹp để tránh chật chội
        cell_pixel_width = 40 if self.grid_size == 9 else 35  # Tăng kích thước ô cho 16x16
        cell_pixel_height = 40 if self.grid_size == 9 else 35
    
        for i in range(self.grid_size):
            row_cells = []
            row_vars = []
        
            for j in range(self.grid_size):
                cell_frame = Frame(self.board_frame, 
                                  bg="black",
                                  width=cell_pixel_width, 
                                  height=cell_pixel_height,
                                  padx=1, pady=1)
                cell_frame.grid(row=i, column=j, padx=1, pady=1)
            
                # Tăng độ dày đường viền giữa các hộp (3x3 hoặc 4x4)
                if i % box_size == 0 and i > 0:
                    cell_frame.grid(pady=(5, 1))  # Đường trên dày hơn
                if j % box_size == 0 and j > 0:
                    cell_frame.grid(padx=(5, 1))  # Đường trái dày hơn
                if i % box_size == 0 and i > 0 and j % box_size == 0 and j > 0:
                    cell_frame.grid(padx=(15, 1), pady=(15, 1))  # Góc dày hơn
            
                cell_var = StringVar()
            
                cell = Entry(cell_frame, width=cell_width, font=("Arial", cell_font_size, "bold"), 
                            justify="center", textvariable=cell_var,
                            borderwidth=0, highlightthickness=1,
                            highlightbackground="#AAAAAA")
            
                cell.pack(fill="both", expand=True)
            
                # Lưu vị trí ô
                cell.position = (i, j)
            
                # Gắn sự kiện
                cell.bind("<FocusIn>", lambda e, pos=(i, j): self._on_cell_focus(pos))
                cell.bind("<KeyRelease>", lambda e, pos=(i, j): self._on_cell_input(e, pos))
            
                row_cells.append(cell)
                row_vars.append(cell_var)
        
            self.cells.append(row_cells)
            self.cell_vars.append(row_vars)
    
        # Điều chỉnh kích thước cửa sổ cho lưới 16x16
        if self.grid_size == 16:
            self.master.geometry("900x850")  # Tăng kích thước cửa sổ cho 16x16
        else:
            self.master.geometry("875x670")  # Kích thước mặc định cho 9x9
    
    def _create_controls(self):
        """Tạo các nút điều khiển"""
        controls_frame = Frame(self.master, bg=self.bg_color, pady=10)
        controls_frame.pack(fill="x")
    
        # Nút trợ giúp
        help_btn = Button(controls_frame, text="Trợ giúp", font=("Arial", 12),
                        command=self._show_instructions, bg="#2980B9", fg="white", width=10)
        help_btn.pack(side="left", padx=10)
    
        # Nút gợi ý
        hint_btn = Button(controls_frame, text="Gợi ý", font=("Arial", 12),
                        command=self._on_hint, bg="#3498DB", fg="white", width=10)
        hint_btn.pack(side="left", padx=10)
    
        # Nút kiểm tra
        check_btn = Button(controls_frame, text="Kiểm tra", font=("Arial", 12),
                        command=self._on_check, bg="#E67E22", fg="white", width=12)
        check_btn.pack(side="left", padx=10)
    
        # Thêm combobox chọn thuật toán giải
        ttk.Label(controls_frame, text="Thuật toán:", font=("Arial", 12), 
                  background=self.bg_color).pack(side="left", padx=10)
    
        self.algorithm_var = StringVar(value="backtracking")
        algorithm_menu = ttk.Combobox(controls_frame, textvariable=self.algorithm_var,
                             values=["dfs", "bfs", "backtracking", "simulated_annealing", "A*", "and_or", "dqn"],
                             state="readonly", width=12)
        algorithm_menu.pack(side="left", padx=5)
    
        # Nút giải
        solve_btn = Button(controls_frame, text="Giải", font=("Arial", 12),
                        command=self._on_solve, bg="#9B59B6", fg="white", width=10)
        solve_btn.pack(side="left", padx=10)
    
        # Nút xóa
        clear_btn = Button(controls_frame, text="Xóa", font=("Arial", 12),
                        command=self._on_clear, bg="#E74C3C", fg="white", width=10)
        clear_btn.pack(side="left", padx=10)
    
        # Nút trò chơi mới
        new_game_btn = Button(controls_frame, text="Trò chơi mới", font=("Arial", 12),
                            command=self._on_new_game, bg="#27AE60", fg="white", width=12)
        new_game_btn.pack(side="left", padx=10)
    
    def _create_status_bar(self):
        """Tạo thanh trạng thái"""
        self.status_frame = Frame(self.master, bg="#DDDDDD", height=25)
        self.status_frame.pack(side="bottom", fill="x")
        
        self.status_var = StringVar(value="Sẵn sàng. Bắt đầu điền số vào các ô trống.")
        self.status_label = Label(self.status_frame, textvariable=self.status_var,
                                font=("Arial", 9), bg="#DDDDDD", fg=self.fg_color,
                                anchor="w", padx=10)
        self.status_label.pack(fill="x")
    
    def _update_timer(self):
        """Cập nhật hiển thị thời gian"""
        if hasattr(self, 'controller') and self.controller.model.game_active:
            elapsed_time = self.controller.model.get_elapsed_time()
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.timer_var.set(f"Thời gian: {minutes:02d}:{seconds:02d}")
        
        # Cập nhật mỗi giây
        self.master.after(1000, self._update_timer)
    
    def _on_back(self):
        """Xử lý khi người dùng nhấn nút quay lại"""
        if hasattr(self, 'controller') and hasattr(self.controller, 'app'):
            if self.confirm_dialog("Quay lại", "Bạn có chắc muốn quay lại màn hình chọn? Tiến trình hiện tại sẽ bị mất."):
                self.controller.app.show_selection_screen()
    
    def _on_new_game(self):
        """Xử lý khi người dùng nhấn nút trò chơi mới"""
        if hasattr(self, 'controller') and hasattr(self.controller, 'app'):
            if self.confirm_dialog("Trò chơi mới", "Bạn có chắc muốn bắt đầu trò chơi mới? Tiến trình hiện tại sẽ bị mất."):
                self.controller.app.show_selection_screen()
    
    def _show_instructions(self):
        """Hiển thị hướng dẫn chơi game"""
        instructions_window = Toplevel(self.master)
        instructions_window.title("Hướng dẫn chơi")
        instructions_window.geometry("500x600")
        instructions_window.resizable(False, False)
        
        # Cấu hình cửa sổ popup
        instructions_window.configure(bg=self.bg_color)
        
        # Làm cho hộp thoại modal
        instructions_window.transient(self.master)
        instructions_window.grab_set()
        
        # Căn giữa cửa sổ
        instructions_window.geometry("+%d+%d" % (
            self.master.winfo_rootx() + self.master.winfo_width()//2 - 250,
            self.master.winfo_rooty() + self.master.winfo_height()//2 - 250
        ))
        
        # Tiêu đề
        Label(instructions_window, text="HƯỚNG DẪN CHƠI", 
             font=("Arial", 14, "bold"),
             bg=self.bg_color, fg=self.fg_color).pack(anchor="center", pady=10)
        
        # Khung nội dung có thanh cuộn
        content_frame = Frame(instructions_window, bg=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Thông tin lưới
        grid_info = "1-9" if self.grid_size == 9 else "1-16"
        box_info = "3×3" if self.grid_size == 9 else "4×4"
        
        instructions_text = (
            f"1. Điền vào lưới sao cho mỗi hàng, cột và hộp {box_info} đều chứa các số {grid_info} không lặp lại.\n\n"
            "2. Nhấp vào ô trống và gõ số để thực hiện nước đi.\n"
            "   Ô màu xám là các ô được điền sẵn và không thể thay đổi.\n"
            "   Ô màu xanh lá cây cho biết số đúng.\n"
            "   Ô màu đỏ cho biết số sai.\n\n"
            "3. Bạn có 3 mạng cho mỗi trò chơi. Mỗi nước đi sai sẽ mất 1 mạng.\n"
            "   Khi hết mạng, trò chơi kết thúc.\n\n"
            "4. Bộ đếm thời gian hiển thị thời gian chơi. Hãy cố gắng giải càng nhanh càng tốt.\n\n"
            "5. Phím tắt:\n"
            f"   Số {grid_info}: Nhập giá trị\n"
            "   Delete hoặc Backspace: Xóa ô\n\n"
            "6. Sử dụng các nút bên dưới bảng:\n"
            "   Trợ giúp: Hiển thị màn hình hướng dẫn này.\n"
            "   Gợi ý: Hiển thị một số đúng trong một ô trống ngẫu nhiên.\n"
            "   Kiểm tra: Kiểm tra tiến trình hiện tại và đánh dấu các lỗi.\n"
            "   Giải: Hiển thị toàn bộ lời giải cho câu đố.\n"
            "   Xóa: Xóa tất cả các đầu vào của người dùng khỏi bảng.\n"
            "   Trò chơi mới: Quay lại màn hình chọn để bắt đầu trò chơi mới.\n"
            "   Quay lại: Quay lại màn hình chọn.\n\n"
        )
        
        instructions_content = Label(content_frame, text=instructions_text, 
                                   font=("Arial", 10), justify="left",
                                   bg=self.bg_color, fg=self.fg_color,
                                   wraplength=460)
        instructions_content.pack(anchor="w", pady=5)
        
        # Mã màu
        legend_frame = Frame(content_frame, bg=self.bg_color, pady=5)
        legend_frame.pack(fill="x")
        
        Label(legend_frame, text="MÃ MÀU:", font=("Arial", 10, "bold"),
             bg=self.bg_color, fg=self.fg_color).pack(anchor="w")
        
        # Tạo mẫu màu với nhãn
        color_samples = [
            ("Ô gốc", self.original_cell_color),
            ("Số đúng", self.success_color),
            ("Số sai", self.error_color),
            ("Gợi ý", self.highlight_color)
        ]
        
        for i, (label_text, color) in enumerate(color_samples):
            sample_frame = Frame(legend_frame, bg=self.bg_color)
            sample_frame.pack(anchor="w", pady=2)
            
            color_box = Frame(sample_frame, bg=color, width=20, height=20)
            color_box.pack(side="left", padx=5)
            
            Label(sample_frame, text=label_text, font=("Arial", 9),
                 bg=self.bg_color, fg=self.fg_color).pack(side="left")
        
        # Nút đóng
        close_button = Button(instructions_window, text="Đóng", font=("Arial", 10, "bold"),
                            command=instructions_window.destroy, bg="#3498DB", fg="white", width=10)
        close_button.pack(pady=15)
    
    # Các phương thức giao diện để kết nối với controller
    def set_controller(self, controller):
        """Thiết lập controller cho view này"""
        self.controller = controller
    
    def update_board(self, board, original_cells):
        """Cập nhật giao diện với trạng thái bảng mới"""
        self.original_cells = original_cells
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.cells[i][j]
                value = board[i][j]
                
                if value == 0:
                    self.cell_vars[i][j].set("")
                    cell.config(bg="white", state="normal")
                else:
                    # Với lưới 16x16, sử dụng chữ cái cho 10-16
                    if self.grid_size == 16 and value > 9:
                        self.cell_vars[i][j].set(chr(ord('A') + value - 10))
                    else:
                        self.cell_vars[i][j].set(str(value))
                    
                    if (i, j) in original_cells:
                        cell.config(bg=self.original_cell_color, state="readonly")
                    else:
                        cell.config(bg="white", state="normal")

    def highlight_cell(self, row, col, color):
        """Tô sáng một ô cụ thể với màu cho trước"""
        if row < self.grid_size and col < self.grid_size:
            self.cells[row][col].config(bg=color)

    def update_lives_display(self, lives):
        """Cập nhật hiển thị mạng"""
        self.lives_var.set(f"Mạng: {lives}")

    def show_message(self, title, message):
        """Hiển thị thông báo"""
        messagebox.showinfo(title, message)
        self.update_status(message)

    def show_error(self, title, message):
        """Hiển thị thông báo lỗi"""
        self.update_status(f"LỖI: {message}", is_error=True)
        
        # Tạo cửa sổ thông báo tự động đóng
        error_window = Toplevel(self.master)
        error_window.title(title)
        error_window.geometry("300x100")
        error_window.resizable(False, False)
        
        # Căn giữa cửa sổ
        error_window.geometry("+%d+%d" % (
            self.master.winfo_rootx() + self.master.winfo_width()//2 - 150,
            self.master.winfo_rooty() + self.master.winfo_height()//2 - 50
        ))
        
        # Thêm thông báo
        Label(error_window, text=message, wraplength=280, pady=10).pack(expand=True)
        
        # Tự động đóng sau 1.5 giây
        error_window.after(1500, error_window.destroy)

    def show_success(self, completion_time=None):
        """Hiển thị thông báo thành công khi giải xong câu đố"""
        if completion_time:
            minutes, seconds = divmod(int(completion_time), 60)
            time_text = f"Thời gian hoàn thành: {minutes:02d}:{seconds:02d}"
            messagebox.showinfo("Chúc mừng!", f"Bạn đã giải thành công câu đố Sudoku!\n\n{time_text}")
            self.update_status(f"Hoàn thành thành công! {time_text}")
        else:
            messagebox.showinfo("Chúc mừng!", "Bạn đã giải thành công câu đố Sudoku!")
            self.update_status("Hoàn thành thành công! Bắt đầu trò chơi mới để chơi tiếp.")

    def show_game_over(self):
        """Hiển thị thông báo kết thúc trò chơi khi hết mạng"""
        messagebox.showinfo("Game Over", "Bạn đã hết mạng! Hãy thử lại.")
        self.update_status("Trò chơi kết thúc! Hết mạng. Bắt đầu trò chơi mới để chơi tiếp.")

    def confirm_dialog(self, title, message):
        """Hiển thị hộp thoại xác nhận"""
        return messagebox.askyesno(title, message)

    def update_status(self, message, is_error=False):
        """Cập nhật thanh trạng thái với thông báo"""
        self.status_var.set(message)
        if is_error:
            self.status_label.config(fg="#CC0000")
        else:
            self.status_label.config(fg=self.fg_color)
        
        # Tự động đặt lại trạng thái sau 3 giây
        self.master.after(3000, lambda: self.status_var.set("Sẵn sàng. Bắt đầu điền số vào các ô trống."))
        self.master.after(3000, lambda: self.status_label.config(fg=self.fg_color))
    
    # Các phương thức xử lý sự kiện
    def _on_cell_focus(self, position):
        """Xử lý khi ô được focus"""
        if not hasattr(self, 'controller'):
            return
            
        row, col = position
        # Tô sáng các ô liên quan
        box_size = 3 if self.grid_size == 9 else 4
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Đặt lại các ô không đặc biệt
                if (i, j) not in self.original_cells and self.controller.model.board[i][j] == 0:
                    self.highlight_cell(i, j, "white")
                
                # Tô sáng các ô liên quan
                if (i == row or j == col or 
                    (i//box_size == row//box_size and j//box_size == col//box_size)) and (i, j) != (row, col):
                    if (i, j) not in self.original_cells and self.controller.model.board[i][j] == 0:
                        self.highlight_cell(i, j, "#F8F8F8")  # Xám rất nhạt

    def _on_cell_input(self, event, position):
        """Xử lý khi người dùng nhập vào ô"""
        if hasattr(self, 'controller'):
            row, col = position
            
            # Bỏ qua nếu đây là ô gốc
            if (row, col) in self.original_cells:
                return
            
            # Lấy giá trị
            value = self.cell_vars[row][col].get()
            
            # Xử lý backspace/delete
            if not value and event.keysym in ('BackSpace', 'Delete'):
                self.controller.clear_cell(row, col)
                return
            
            # Với lưới 16x16, xử lý chữ cái A-G cho giá trị 10-16
            if self.grid_size == 16 and value.upper() in "ABCDEFG":
                # Chuyển đổi A-G thành 10-16
                num_val = ord(value.upper()) - ord('A') + 10
                self.cell_vars[row][col].set(value.upper())
                self.controller.make_move(row, col, num_val)
                return
            
            # Lọc đầu vào không phải số
            if not value.isdigit() or int(value) == 0:
                self.controller.clear_cell(row, col)
                return
            
            # Kiểm tra giới hạn số dựa vào kích thước lưới
            max_value = self.grid_size
            if int(value) > max_value:
                # Nếu người dùng nhập số lớn hơn kích thước lưới, chỉ lấy chữ số cuối cùng
                value = value[-1]
                self.cell_vars[row][col].set(value)
            
            # Thực hiện nước đi
            self.controller.make_move(row, col, int(value))
    
    def _on_hint(self):
        """Xử lý khi người dùng nhấn nút gợi ý"""
        if hasattr(self, 'controller'):
            self.controller.get_hint()
    
    def _on_check(self):
        """Xử lý khi người dùng nhấn nút kiểm tra"""
        if hasattr(self, 'controller'):
            self.controller.check_solution()
    
    def _on_solve(self):
        """Xử lý khi người dùng nhấn nút giải"""
        if not hasattr(self, 'controller'):
            return
    
        # Lấy thuật toán được chọn
        algorithm = self.algorithm_var.get()
    
        # Gọi phương thức giải trong controller
        self.controller.solve_with_algorithm(algorithm)
    
    def _on_clear(self):
        """Xử lý khi người dùng nhấn nút xóa"""
        if hasattr(self, 'controller'):
            self.controller.clear_board()

    def show_algorithm_comparison(self, metrics):
        """
        Hiển thị kết quả so sánh thuật toán.
    
        Args:
            metrics: Từ điển chứa các thông số hiệu suất của thuật toán
        """
        # Tạo cửa sổ mới
        comparison_window = Toplevel(self.master)
        comparison_window.title("Kết quả thuật toán")
        comparison_window.geometry("800x600")
        comparison_window.resizable(True, True)
    
        # Cấu hình của sổ
        comparison_window.configure(bg=self.bg_color)
    
        # Tạo frame chứa thông tin
        info_frame = Frame(comparison_window, bg=self.bg_color, padx=20, pady=20)
        info_frame.pack(fill="both", expand=True)
    
        # Hiển thị tên thuật toán
        algorithm_name = self.algorithm_var.get()
        Label(info_frame, text=f"Thuật toán: {algorithm_name}", 
              font=("Arial", 16, "bold"), bg=self.bg_color, fg=self.fg_color).pack(pady=10)
    
        # Hiển thị kết quả
        result_text = "Thành công" if metrics['is_solved'] else "Không tìm thấy lời giải"
        result_color = "#27AE60" if metrics['is_solved'] else "#E74C3C"
        Label(info_frame, text=f"Kết quả: {result_text}", 
              font=("Arial", 14, "bold"), bg=self.bg_color, fg=result_color).pack(pady=5)
    
        # Hiển thị các thông số
        metrics_frame = Frame(info_frame, bg=self.bg_color)
        metrics_frame.pack(fill="x", pady=10)
    
        # Tạo bảng thông số
        columns = ("Thông số", "Giá trị")
        metrics_table = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=6)
        metrics_table.heading("Thông số", text="Thông số")
        metrics_table.heading("Giá trị", text="Giá trị")
        metrics_table.column("Thông số", width=300)
        metrics_table.column("Giá trị", width=300)
    
        # Thêm dữ liệu vào bảng
        metrics_table.insert("", "end", values=("Thời gian thực thi (giây)", f"{metrics['execution_time']:.6f}"))
        metrics_table.insert("", "end", values=("Số trạng thái đã khám phá", f"{metrics['states_explored']}"))
        metrics_table.insert("", "end", values=("Số trạng thái tối đa trong bộ nhớ", f"{metrics['max_states_in_memory']}"))
        metrics_table.insert("", "end", values=("Giá trị heuristic h(n)", f"{metrics['h_value']}"))
        metrics_table.insert("", "end", values=("Số bước thực hiện g(n)", f"{metrics['g_value']}"))
        metrics_table.insert("", "end", values=("Tổng chi phí f(n) = g(n) + h(n)", f"{metrics['f_value']}"))
    
        metrics_table.pack(fill="x", padx=10, pady=10)
    
        # Tạo biểu đồ so sánh
        chart_frame = Frame(info_frame, bg=self.bg_color)
        chart_frame.pack(fill="both", expand=True, pady=10)
    
        # Tạo biểu đồ cột cho thời gian và số trạng thái
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
        # Biểu đồ thời gian
        ax1.bar(['Thời gian (s)'], [metrics['execution_time']], color='#3498DB')
        ax1.set_title('Thời gian thực thi')
        ax1.set_ylabel('Giây')
    
        # Biểu đồ số trạng thái
        ax2.bar(['Trạng thái đã khám phá', 'Trạng thái tối đa trong bộ nhớ'], 
                [metrics['states_explored'], metrics['max_states_in_memory']], 
                color=['#9B59B6', '#E67E22'])
        ax2.set_title('Không gian trạng thái')
        ax2.set_ylabel('Số lượng')
    
        # Hiển thị biểu đồ trong tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
        # Thêm nút đóng
        Button(comparison_window, text="Đóng", font=("Arial", 12, "bold"),
               command=comparison_window.destroy, bg="#3498DB", fg="white", width=10).pack(pady=15)


