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
