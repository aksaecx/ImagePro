# ======================== BAGIAN 1 (Kontributor 1) ========================
# Import libraries dan Class Definition (1-100)
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
        self.root.configure(bg="#f0f0f0")
        
        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background="#f0f0f0")
        self.style.configure('TLabel', background="#f0f0f0", font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground="#2c3e50")
        self.style.configure('Section.TLabelframe', font=('Helvetica', 11, 'bold'), borderwidth=2, relief="solid")
        self.style.configure('Section.TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        self.style.map('TButton', 
                      foreground=[('active', 'white'), ('!active', 'black')],
                      background=[('active', '#3498db'), ('!active', '#ecf0f1')])
        
        # Variables
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="IMAGE PROCESSING PRO", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Professional Digital Image Processing Tool", font=('Helvetica', 12))
        subtitle_label.pack()
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text=" Image Input ", style='Section.TLabelframe')
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        btn_load = ttk.Button(input_frame, text="Load Image", command=self.load_image, style='Accent.TButton')
        btn_load.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        self.file_label = ttk.Label(input_frame, text="No image selected", font=('Helvetica', 9))
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
            
        def brightness_decrease():
            result = cv2.subtract(self.original_image, np.ones(self.original_image.shape, dtype=np.uint8) * 50)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        def multiply():
            result = cv2.multiply(self.original_image, np.ones(self.original_image.shape, dtype=np.uint8) * 1.5)
            self.processed_image = result
            self.display_image(result, self.processed_canvas)
            dialog.destroy()
            
        ttk.Button(dialog, text="Increase Brightness (+50)", command=brightness_increase).pack(pady=5)
        ttk.Button(dialog, text="Decrease Brightness (-50)", command=brightness_decrease).pack(pady=5)
        ttk.Button(dialog, text="Multiply (×1.5)", command=multiply).pack(pady=5)
