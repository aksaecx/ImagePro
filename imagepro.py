# ======================== BAGIAN 1 TAMPILAN (Kontributor 1) ========================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImagePro:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Pro - Professional Image Processing")
        self.root.geometry("1280x800")
        self.root.configure(bg="#1e1e2d")  # Dark background
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.bg_color = "#1e1e2d"
        self.card_color = "#2a2a3a"
        self.accent_color = "#4e8cff"
        self.text_color = "#ffffff"
        self.secondary_text = "#b0b0b0"
        
        self.style.configure('.', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, 
                           font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8,
                           background="#3a3a4a", foreground=self.text_color,
                           borderwidth=0)
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), 
                           foreground=self.accent_color)
        self.style.configure('Section.TLabelframe', font=('Segoe UI', 12, 'bold'), 
                           borderwidth=0, relief="flat", background=self.bg_color,
                           foreground=self.accent_color)
        self.style.configure('Section.TLabelframe.Label', font=('Segoe UI', 12, 'bold'),
                           foreground=self.accent_color)
        self.style.configure('Accent.TButton', background=self.accent_color,
                           foreground="#ffffff")
        self.style.configure('TEntry', fieldbackground=self.card_color,
                           foreground=self.text_color, insertcolor=self.text_color)
        
        self.style.map('TButton',
                      background=[('active', '#4a4a5a'), ('!active', '#3a3a4a')],
                      foreground=[('active', self.text_color), ('!active', self.text_color)])
        self.style.map('Accent.TButton',
                      background=[('active', '#3d7ae5'), ('!active', self.accent_color)],
                      foreground=[('active', '#ffffff'), ('!active', '#ffffff')])
        
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_frame = ttk.Frame(main_frame, style='TFrame')
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        title_label = ttk.Label(title_frame, 
                              text="IMAGE PRO", 
                              style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, 
                                 text="By : Evaleona - Aksa - Rifai - Safri", 
                                 font=('Segoe UI', 12), 
                                 foreground=self.secondary_text)
        subtitle_label.pack(pady=(5, 0))
        
        input_frame = ttk.LabelFrame(main_frame, 
                                   text=" IMAGE INPUT ", 
                                   style='Section.TLabelframe',
                                   padding=(15, 10))
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        inner_input_frame = ttk.Frame(input_frame, style='TFrame')
        inner_input_frame.pack(fill=tk.BOTH, expand=True)
        
        btn_load = ttk.Button(inner_input_frame, 
                             text="📁 Load Image", 
                             command=self.load_image, 
                             style='Accent.TButton')
        btn_load.grid(row=0, column=0, padx=(0, 15), pady=5)
        
        self.file_label = ttk.Label(inner_input_frame, 
                                  text="No image selected", 
                                  font=('Segoe UI', 9),
                                  foreground=self.secondary_text)
        self.file_label.grid(row=0, column=1, sticky=tk.W)
        
# ======================== BAGIAN 2 (Kontributor 2) ========================
# GUI Setup dan Image Loading (101-200)
        # Processing section
        process_frame = ttk.LabelFrame(main_frame, text=" Processing Tools ", style='Section.TLabelframe')
        process_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Basic Operations
        basic_frame = ttk.LabelFrame(process_frame, text="Basic Operations")
        basic_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        btn_gray = ttk.Button(basic_frame, text="Grayscale", command=self.convert_grayscale)
        btn_gray.grid(row=0, column=0, padx=5, pady=5)
        
        btn_binary = ttk.Button(basic_frame, text="Binary", command=self.convert_binary)
        btn_binary.grid(row=0, column=1, padx=5, pady=5)
        
        btn_arith = ttk.Button(basic_frame, text="Arithmetic", command=self.arithmetic_operations)
        btn_arith.grid(row=0, column=2, padx=5, pady=5)
        
        btn_logic = ttk.Button(basic_frame, text="Logic", command=self.logic_operations)
        btn_logic.grid(row=0, column=3, padx=5, pady=5)
        
        # Advanced Operations
        adv_frame = ttk.LabelFrame(process_frame, text="Advanced Operations")
        adv_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        btn_hist = ttk.Button(adv_frame, text="Histogram", command=self.show_histogram)
        btn_hist.grid(row=0, column=0, padx=5, pady=5)
        
        btn_conv = ttk.Button(adv_frame, text="Convolution", command=self.convolution_filter)
        btn_conv.grid(row=0, column=1, padx=5, pady=5)
        
        btn_morph = ttk.Button(adv_frame, text="Morphology", command=self.morphology_operations)
        btn_morph.grid(row=0, column=2, padx=5, pady=5)
        
        btn_reset = ttk.Button(adv_frame, text="Reset", command=self.reset_image)
        btn_reset.grid(row=0, column=3, padx=5, pady=5)
        
        # Display section
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        original_frame = ttk.LabelFrame(display_frame, text=" Original Image ", style='Section.TLabelframe')
        original_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.original_canvas = tk.Canvas(original_frame, width=400, height=400, bg="white", bd=2, relief="groove")
        self.original_canvas.pack(padx=10, pady=10)
        
        # Processed image
        processed_frame = ttk.LabelFrame(display_frame, text=" Processed Image ", style='Section.TLabelframe')
        processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.processed_canvas = tk.Canvas(processed_frame, width=400, height=400, bg="white", bd=2, relief="groove")
        self.processed_canvas.pack(padx=10, pady=10)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Custom style for accent button
        self.style.configure('Accent.TButton', 
                           font=('Helvetica', 10, 'bold'),
                           foreground='white',
                           background='#3498db',
                           padding=8)
        self.style.map('Accent.TButton',
                     foreground=[('active', 'white'), ('!active', 'white')],
                     background=[('active', '#2980b9'), ('!active', '#3498db')])
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.current_image = self.original_image.copy()
                short_path = file_path.split('/')[-1][:30] + "..." if len(file_path.split('/')[-1]) > 30 else file_path.split('/')[-1]
                self.file_label.config(text=f"Loaded: {short_path}")
                self.display_image(self.original_image, self.original_canvas)
                self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# ======================== BAGIAN 2 (Kontributor 2) ========================
# GUI Setup dan Image Loading (101-200)
        # Processing section
        process_frame = ttk.LabelFrame(main_frame, text=" Processing Tools ", style='Section.TLabelframe')
        process_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Basic Operations
        basic_frame = ttk.LabelFrame(process_frame, text="Basic Operations")
        basic_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        btn_gray = ttk.Button(basic_frame, text="Grayscale", command=self.convert_grayscale)
        btn_gray.grid(row=0, column=0, padx=5, pady=5)
        
        btn_binary = ttk.Button(basic_frame, text="Binary", command=self.convert_binary)
        btn_binary.grid(row=0, column=1, padx=5, pady=5)
        
        btn_arith = ttk.Button(basic_frame, text="Arithmetic", command=self.arithmetic_operations)
        btn_arith.grid(row=0, column=2, padx=5, pady=5)
        
        btn_logic = ttk.Button(basic_frame, text="Logic", command=self.logic_operations)
        btn_logic.grid(row=0, column=3, padx=5, pady=5)
        
        # Advanced Operations
        adv_frame = ttk.LabelFrame(process_frame, text="Advanced Operations")
        adv_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        btn_hist = ttk.Button(adv_frame, text="Histogram", command=self.show_histogram)
        btn_hist.grid(row=0, column=0, padx=5, pady=5)
        
        btn_conv = ttk.Button(adv_frame, text="Convolution", command=self.convolution_filter)
        btn_conv.grid(row=0, column=1, padx=5, pady=5)
        
        btn_morph = ttk.Button(adv_frame, text="Morphology", command=self.morphology_operations)
        btn_morph.grid(row=0, column=2, padx=5, pady=5)
        
        btn_reset = ttk.Button(adv_frame, text="Reset", command=self.reset_image)
        btn_reset.grid(row=0, column=3, padx=5, pady=5)
        
        # Display section
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        original_frame = ttk.LabelFrame(display_frame, text=" Original Image ", style='Section.TLabelframe')
        original_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.original_canvas = tk.Canvas(original_frame, width=400, height=400, bg="white", bd=2, relief="groove")
        self.original_canvas.pack(padx=10, pady=10)
        
        # Processed image
        processed_frame = ttk.LabelFrame(display_frame, text=" Processed Image ", style='Section.TLabelframe')
        processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.processed_canvas = tk.Canvas(processed_frame, width=400, height=400, bg="white", bd=2, relief="groove")
        self.processed_canvas.pack(padx=10, pady=10)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Custom style for accent button
        self.style.configure('Accent.TButton', 
                           font=('Helvetica', 10, 'bold'),
                           foreground='white',
                           background='#3498db',
                           padding=8)
        self.style.map('Accent.TButton',
                     foreground=[('active', 'white'), ('!active', 'white')],
                     background=[('active', '#2980b9'), ('!active', '#3498db')])
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.current_image = self.original_image.copy()
                short_path = file_path.split('/')[-1][:30] + "..." if len(file_path.split('/')[-1]) > 30 else file_path.split('/')[-1]
                self.file_label.config(text=f"Loaded: {short_path}")
                self.display_image(self.original_image, self.original_canvas)
                self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# ======================== BAGIAN 3 (Kontributor 3) ========================
# Basic Image Processing (201-300)
    def display_image(self, image, canvas):
        """Display image on canvas with better visualization"""
        if image is None:
            return
            
        # Convert BGR to RGB for display
        if len(image.shape) == 3:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
            
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_w = canvas.winfo_width() - 20
        canvas_h = canvas.winfo_height() - 20
        
        h, w = display_img.shape[:2]
        ratio = min(canvas_w/w, canvas_h/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
            
        resized_img = cv2.resize(display_img, (new_w, new_h))
        
        # Convert to PIL Image and then to PhotoImage
        pil_img = Image.fromarray(resized_img)
        photo = ImageTk.PhotoImage(pil_img)
        
        # Clear canvas and display new image
        canvas.delete("all")
        canvas.create_image(canvas_w//2, canvas_h//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference
        
    def convert_grayscale(self):
        """Convert image to grayscale"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = gray_image
        self.display_image(gray_image, self.processed_canvas)
        
    def convert_binary(self):
        """Convert image to binary using Otsu's method"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Convert to grayscale first
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.processed_image = binary_image
        self.display_image(binary_image, self.processed_canvas)
        
    def arithmetic_operations(self):
        """Perform arithmetic operations"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Create a simple dialog for arithmetic operations
        dialog = tk.Toplevel(self.root)
        dialog.title("Arithmetic Operations")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Select arithmetic operation:").pack(pady=10)
        
        def brightness_increase():
            result = cv2.add(self.original_image, np.ones(self.original_image.shape, dtype=np.uint8) * 50)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
                       
        ttk.Button(dialog, text="Increase Brightness (+50)", command=brightness_increase).pack(pady=5)

# ======================== BAGIAN 4 (Kontributor 4) ========================
# Advanced Image Processing dan Main Function (301-end)
    def logic_operations(self):
        """Perform logic operations"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Create a simple dialog for logic operations
        dialog = tk.Toplevel(self.root)
        dialog.title("Logic Operations")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Select logic operation:").pack(pady=10)
        
        def bitwise_and():
            mask = np.ones(self.original_image.shape, dtype=np.uint8) * 200
            result = cv2.bitwise_and(self.original_image, mask)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        #def bitwise_or():
            #mask = np.ones(self.original_image.shape, dtype=np.uint8) * 50
            #result = cv2.bitwise_or(self.original_image, mask)
            #self.processed_image = result
            #self.display_image(result, self.processed_canvas)
            #dialog.destroy()
            
        #def bitwise_not():
            #result = cv2.bitwise_not(self.original_image)
            #self.processed_image = result
            #self.display_image(result, self.processed_canvas)
            #dialog.destroy()
            
        ttk.Button(dialog, text="AND Operation", command=bitwise_and).pack(pady=5)
        #ttk.Button(dialog, text="OR Operation", command=bitwise_or).pack(pady=5)
        #ttk.Button(dialog, text="NOT Operation", command=bitwise_not).pack(pady=5)
        
    def show_histogram(self):
        """Show histogram of the image"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Create new window for histogram
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Histogram")
        hist_window.geometry("600x400")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if len(self.original_image.shape) == 3:
            # Color image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, label=f'Channel {color.upper()}')
            ax.set_title('RGB Histogram')
        else:
            # Grayscale image
            hist = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
            ax.set_title('Grayscale Histogram')
            
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def convolution_filter(self):
        """Apply convolution filters"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Create dialog for filter selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Convolution Filters")
        dialog.geometry("300x250")
        
        ttk.Label(dialog, text="Select convolution filter:").pack(pady=10)
        
        def apply_blur():
            kernel = np.ones((5, 5), np.float32) / 25  # Mean filter
            result = cv2.filter2D(self.original_image, -1, kernel)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        def apply_sharpen():
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening filter
            result = cv2.filter2D(self.original_image, -1, kernel)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        def apply_edge_sobel():
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = np.sqrt(sobelx*2 + sobely*2)
            result = np.uint8(result)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        def apply_edge_prewitt():
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            result = np.sqrt(prewittx*2 + prewitty*2)
            result = np.uint8(result)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        ttk.Button(dialog, text="Mean Filter (Blur)", command=apply_blur).pack(pady=5)
        ttk.Button(dialog, text="Sharpening Filter", command=apply_sharpen).pack(pady=5)
        ttk.Button(dialog, text="Edge Detection (Sobel)", command=apply_edge_sobel).pack(pady=5)
        ttk.Button(dialog, text="Edge Detection (Prewitt)", command=apply_edge_prewitt).pack(pady=5)
        
    def morphology_operations(self):
        """Perform morphological operations"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Convert to binary first
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create dialog for morphology operations
        dialog = tk.Toplevel(self.root)
        dialog.title("Morphological Operations")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Select morphological operation:").pack(pady=10)
        ttk.Label(dialog, text="(Image will be converted to binary first)").pack()
        
        def apply_erosion_rect():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.erode(binary, kernel, iterations=1)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        def apply_erosion_ellipse():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.erode(binary, kernel, iterations=1)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        #def apply_dilation_rect():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.dilate(binary, kernel, iterations=1)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        #def apply_dilation_ellipse():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.dilate(binary, kernel, iterations=1)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        #def apply_opening():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        #def apply_closing():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        ttk.Label(dialog, text="Erosion:", font=("Arial", 9, "bold")).pack(pady=(10, 0))
        ttk.Button(dialog, text="Erosion (Rectangular SE)", command=apply_erosion_rect).pack(pady=2)
        ttk.Button(dialog, text="Erosion ", command=apply_erosion_ellipse).pack(pady=2)
        
        #ttk.Label(dialog, text="Dilation:", font=("Arial", 9, "bold")).pack(pady=(10, 0))
        #ttk.Button(dialog, text="Dilation (Rectangular SE)", command=apply_dilation_rect).pack(pady=2)
        #ttk.Button(dialog, text="Dilation (Ellipse SE)", command=apply_dilation_ellipse).pack(pady=2)
        
        #ttk.Label(dialog, text="Combination:", font=("Arial", 9, "bold")).pack(pady=(10, 0))
        #ttk.Button(dialog, text="Opening", command=apply_opening).pack(pady=2)
        #ttk.Button(dialog, text="Closing", command=apply_closing).pack(pady=2)
        
    def reset_image(self):
        """Reset to original image"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        self.processed_image = None
        self.processed_canvas.delete("all")
        self.processed_canvas.create_text(200, 200, text="Image has been reset", anchor=tk.CENTER)

def main():
    root = tk.Tk()
    app = ImagePro(root)
    root.mainloop()

if __name__ == "__main__":
    main()
   
