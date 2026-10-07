Title of the Course  : Programming AI Accelerators 
Name of the Faculty  : Girish Varma, P J Narayanan 
Name of the Program : B.Tech./MS/PhD in CS/ECE 
Course Code   : CS3.406 
Semester, Year  : Monsoon 2026 

Course Outcomes 

Upon successful completion of the course, students will be able to: 
CO1: Analyse GPU execution behaviour, memory hierarchy, and parallel programming 
models to evaluate the performance characteristics of deep learning workloads. 
CO2: Design and implement tensor abstractions and runtime systems using modern C++ 
and CUDA for efficient management of multidimensional data. 
CO3: Develop computational graph frameworks and reverse-mode automatic 
differentiation engines to support gradient-based learning. 
CO4: Construct and optimize neural network training pipelines including loss functions, 
optimization algorithms, and numerical stability mechanisms. 
CO5: Design high-performance GPU kernels for matrix multiplication, normalization, and 
attention operations using memory-aware optimization techniques. 
CO6: Develop a complete Transformer-based inference framework capable of loading 
pretrained weights, managing KV caches, and generating text using custom CUDA 
kernels. 
 
Course Topics 

Module I: GPU Architecture and Performance Engineering 
Introduction to Bare-Metal AI Systems, GPU Architecture Fundamentals, CUDA 
Programming Model, GPU Memory Hierarchy, Memory Optimization Techniques, GPU 
Profiling and Performance Analysis, GPU Design Patterns, Tensor Mathematics and 
Memory Layouts 

Module II: Tensor Runtime and Neural Network Training Systems 
Tensor Runtime Design in Modern C++, Tensor Operations and Shape Inference, Reverse
Mode Automatic Differentiation, Dynamic Computational Graphs, Backpropagation 
Engine Design, Neural Networks and Matrix Calculus, Loss Functions and Numerical 
Stability, Optimization Algorithms (SGD, Momentum, AdamW) 

Module III: High-Performance GPU Kernels for Deep Learning
Matrix Multiplication Fundamentals, Shared-Memory Tiled GEMM, Kernel Fusion 
Techniques, Runtime Optimization Strategies, Softmax and Reduction Kernels, 
LayerNorm and RMSNorm, Attention Kernel Engineering, CUDA Graphs and Execution 
Optimization 

Module IV: Transformer Systems and LLM Inference 
Language Modelling Foundations, Tokenization and Embeddings, Self-Attention 
Mathematics, Multi-Head Attention, Transformer Block Architecture, Rotary Positional 
Embeddings (RoPE), KV Cache Management, Checkpoint Loading and Weight 
Serialization, Autoregressive Inference, LLM Deployment and Optimization 

Teaching-Learning Strategies in Brief 

The course follows a project-driven and systems-oriented pedagogy in which students 
progressively build a deep learning framework from first principles using C++ and CUDA. 
Concepts from GPU architecture, numerical optimization, machine learning, and software 
systems are introduced through interactive lectures and reinforced through structured 
laboratory exercises. Every two lectures are followed by a hands-on laboratory session 
that contributes directly to the semester-long MiniTensor-LM project. Assessment 
emphasizes implementation, profiling, debugging, and oral code defense to ensure deep 
understanding and discourage dependence on AI-generated code. By the end of the 
course, students develop an end-to-end Transformer inference system capable of loading 
pretrained weights and generating text using their own runtime.
