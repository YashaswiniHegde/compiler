# Let’s say, we have a processor with 4 cores and one Global Memory(Size: 1GB). Each core contains 4 Matmul Units and 1 Local Memory Unit(Size: 512KB). Each matmul unit can do a matrix multiplication of 32x32 floats independently. Instructions supported are:

# ⁠ cp_global_to_local <src_slice>, <core_num>, <local_dst_slice> ⁠   — 2D copy of the data from Global memory to local memory with strides
# ⁠ cp_local_to_global <core_num>, <local_src_slice> <dst_slice> ⁠.  —  2D copy of the data from Local memory to Global Memory.
# ⁠ matmul <core_num>, <matmul_unit_num>,  <local_slice_A> <local_slice_B> <local_slice_C>. accumulator=True / False ⁠ — Picks the data from the Local memory slices, does the matrix multiplication of 32x32 A and B matrices and stores it to the C matrix slice mentioned. Accumulator indicates whether to “ADD” the data already present in <local_slice_C>


# ⁠ slice ⁠ contains the following information:

# base address of the memory
# start_offset: end_offset : stride  for Dim 1.  (concept similar to python ranges)
# start_offset: end_offset : stride  for Dim 0. — Contiguous dimension ﻿ ﻿ ﻿

# For example, if A is 32x32 float matrix and B is 32x32 float matrix, then C will be 32x32 matrix. Assuming that they will be in designated locations of Global memory, we can do matmul using following instructions

# cp_global_to_local <A, [0:32:1, 0:32:1]> , core=0, <0 /local_mem/, [0:32:1], [0:32:1]>. // Copies 32 x 32 x 4 (float size) = 4096 bytes from base address “A” to Local memory Address 0 in Core 0.
# cp_global_to_local <B, [0:32:1, 0:32:1]> , core=0, <4096, /local_mem/, [0:32:1], [0:32:1]>
# matmul core=0, matmul_unit=0, <0 /local_mem/, [0:32:1], [0:32:1]>, <4096 /local_mem/, [0:32:1], [0:32:1]>, <8192 /local_mem/, [0:32:1], [0:32:1]>.  // Writes the matmul output of core 0, matmul unit 0 to the 8192 offset of local memory
# cp_local_to_global <8192 /local_mem/, [0:32:1], [0:32:1]>,  <C, [0:32:1, 0:32:1]> // Write the output to Global memory with base address C.

# The assignment is to write a Python based tool to generate the optimal instruction set for any Matrix Multiplication workload. Some simple dimensions to start with:

# A = 32x32, B = 32x32
# A = 64x32, B = 32x32
# A = 64x64, B = 64x64

def matmul_with_slice():
    try:
        M = int(input("Enter number of rows for Matrix A: "))
        K = int(input("Enter number of columns for Matrix A: "))

        Kb = int(input("Enter number of rows for Matrix B: "))
        N = int(input("Enter number of columns for Matrix B: "))
        
        assert K == Kb, "Incompatible matrix sizes"

    except ValueError:
        print("Invalid size input. Please enter valid matrix sizes")
        return

    slice_size = 32
    A_offset = ((slice_size) * (slice_size) * 4)
    B_offset = A_offset + ((slice_size) * (slice_size) * 4)
    #print(f"A_offset:{A_offset}\n")
    #print(f"B_offset:{B_offset}\n")
    cnt = 0
    for i in range(0, M, slice_size):
        for j in range(0, N, slice_size):
            for k in range(0, K, slice_size):
                #A_src_slice = A[i:i+slice_size, k:k+slice_size]
                #B_src_slice = B[k:k+slice_size, j:j+slice_size]
                
                #print("cp_global_to_local <A, [i:i+slice_size:1, k:k+slice_size:1]> , core=0, <0 /local_mem/, [i:i+slice_size:1], [k:k+slice_size:1]>")   
                print(f"\ncp_global_to_local <A, [{i}:{i+slice_size}:1, {k}:{k+slice_size}:1]>, core=0, <0 /local_mem/, [{i}:{i+slice_size}:1], [{k}:{k+slice_size}:1]>\n")
                print(f"cp_global_to_local <B, [{k}:{k+slice_size}:1, {j}:{j+slice_size}:1]> , core=0, <0 /local_mem/, [{k}:{k+slice_size}:1], [{j}:{j+slice_size}:1]>\n")   

                
                print(f"matmul core=0, matmul_unit=0, <0 /local_mem/, [{i}:{i+slice_size}:1], [{k}:{k+slice_size}:1]>, < {A_offset} /local_mem/, [{k}:{k+slice_size}:1], [{j}:{j+slice_size}:1]>, < {B_offset} /local_mem/, [{i}:{i+slice_size}:1], [{j}:{j+slice_size}:1]>, accumulator=True\n")
                print(f"cp_local_to_global <{B_offset} /local_mem/, [{i}:{i+slice_size}:1], [{j}:{j+slice_size}:1]>,  <C, [{i}:{i+slice_size}:1, {j}:{j+slice_size}:1]>\n")
                cnt=cnt+1  
                #C[i:i+slize_size, j:j+slice_size] += A_src_slice * B_src_slice
                print(f"cnt= {cnt}")


def multicore_mmunit_matmul_with_slice():
    try:
        M = int(input("Enter number of rows for Matrix A: "))
        K = int(input("Enter number of columns for Matrix A: "))

        Kb = int(input("Enter number of rows for Matrix B: "))
        N = int(input("Enter number of columns for Matrix B: "))
        
        assert K == Kb, "Incompatible matrix sizes"

    except ValueError:
        print("Invalid size input. Please enter valid matrix sizes")
        return

    NUM_CORE = 4
    NUM_MMUNIT = 4
    slice_size = 32
    slicesize_bytes = ((slice_size) * (slice_size) * 4) # 4096, 4 bytes for the float 
    C_offset = 512000 - 4096

    ra = M // slice_size
    rb = K // slice_size
    cb = N // slice_size

    A_core_mmunit_memloc_flags = {}  #A_core_mmunit_memloc_flags[(0,0:32,32:64)] = 4096 , defining which core and slice of A is accupied the local memory
    B_core_mmunit_memloc_flags = {}  #B_core_mmunit_matslice_locmem_flags[(0,0:32,0:32)] = 8192 , defining which core and slice of B is accupied the local memory

    core_flag = {}  #core_flag[01]=True ,indicating core number 01 is full.
    mm_flag = {}    #mm_flag[01]=True ,indicating Matmul unit 01 is full.

    C_org = [[0 for _ in range(slice_size)] for _ in range(slice_size)] # Initialize C array to zero

    num_mult = ra * rb * cb #num of multiplications 
    corecnt=0
    mmunit_num = 0
    #if core_flag.get(corecnt,False) is False:

    for mmunit_i in range(0, ra): #Each Core has 4 Matmul units
        for mmunit_j in range(0, cb):
    
            #C[0:32 , 0:32] = 0
            C = C_org
            print("\nReset the C array\n")
            print(f"\ncp_global_to_local <C, [{0}:{slice_size}:1, {0}:{slice_size}:1]>, core={corecnt}, <{C_offset} /local_mem/, [{0}:{slice_size}:1], [{0}:{slice_size}:1]>\n")
    
            for mmunit_k in range(0, rb):

                print(f"Core number: {corecnt}, Matmul unit number: {mmunit_num}\n")

                A_offset = (2*mmunit_num) * slicesize_bytes 
                B_offset = (2*mmunit_num+1) * slicesize_bytes
                    
                A_row_range = mmunit_i*slice_size
                A_col_range = mmunit_k*slice_size

                B_row_range = mmunit_k * slice_size
                B_col_range = mmunit_j * slice_size

                C_row_range = mmunit_i * slice_size
                C_col_range = mmunit_j * slice_size

                #print(f"A_offset: {A_core_mmunit_memloc_flags.get((corecnt, A_row_range, A_row_range+slice_size, A_col_range, A_col_range+slice_size))}")

                if A_core_mmunit_memloc_flags.get((corecnt, A_row_range, A_row_range+slice_size, A_col_range, A_col_range+slice_size)) is None:
                    print("{")
                    print(f"\ncp_global_to_local <A, [{A_row_range}:{A_row_range+slice_size}:1, {A_col_range}:{A_col_range+slice_size}:1]>, core={corecnt}, <{A_offset} /local_mem/, [{A_row_range}:{A_row_range+slice_size}:1], [{A_col_range}:{A_col_range+slice_size}:1]>\n")
                    A_core_mmunit_memloc_flags[(corecnt, A_row_range, A_row_range+slice_size, A_col_range, A_col_range+slice_size)] = A_offset 
                else:
                    A_offset = A_core_mmunit_memloc_flags.get((corecnt, A_row_range, A_row_range+slice_size, A_col_range, A_col_range+slice_size)) 
                    print(f"A is already available in the local memory and its offset is {A_offset}\n")
                    print("{")

                if B_core_mmunit_memloc_flags.get((corecnt, B_row_range, B_row_range+slice_size, B_col_range, B_col_range+slice_size)) is None:
                    print(f"\ncp_global_to_local <B, [{B_row_range}:{B_row_range+slice_size}:1, {B_col_range}:{B_col_range+slice_size}:1]>, core={corecnt}, <{B_offset} /local_mem/, [{B_row_range}:{B_row_range+slice_size}:1], [{B_col_range}:{B_col_range+slice_size}:1]>\n")
                    B_core_mmunit_memloc_flags[(corecnt, B_row_range, B_row_range+slice_size, B_col_range, B_col_range+slice_size)] = B_offset 
                    print("}\n")
                else:
                    B_offset = B_core_mmunit_memloc_flags.get((corecnt, B_row_range, B_row_range+slice_size, B_col_range, B_col_range+slice_size))
                    print(f"B is already available in the local memory and its offset is {B_offset}\n")
                    print("}\n")

                print(f"\nmatmul core={corecnt}, matmul_unit={mmunit_num}, <{A_offset} /local_mem/, [{A_row_range}:{A_row_range+slice_size}:1], [{A_col_range}:{A_col_range+slice_size}:1]>, < {B_offset} /local_mem/, [{B_row_range}:{B_row_range+slice_size}:1], [{B_col_range}:{B_col_range+slice_size}:1]>, < {C_offset} /local_mem/, [{C_row_range}:{C_row_range+slice_size}:1], [{C_col_range}:{C_col_range+slice_size}:1]>, accumulator=True\n")

                print(f"\ncp_local_to_global <{C_offset} /local_mem/, [{C_row_range}:{C_row_range+slice_size}:1], [{C_col_range}:{C_col_range+slice_size}:1]>,  <C, [{C_row_range}:{C_row_range+slice_size}:1, {C_col_range}:{C_col_range+slice_size}:1]>\n")

                mmunit_num+=1
                mmunit_num=mmunit_num % NUM_MMUNIT
                if mmunit_num == 0: corecnt+=1


# Run the program
if __name__ == "__main__":
    #matmul_with_slice()
    multicore_mmunit_matmul_with_slice()


