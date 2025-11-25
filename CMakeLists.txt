cmake_minimum_required(VERSION 3.21)
project(matmul_4070ti)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

if(CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "--device-debug -O0 -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

if(MSVC)
	string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	string(REPLACE "/RTCsu" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /MDd")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /MD")

	set(CUDA_NVCC_FLAGS_DEBUG "--device-debug -O0 -g -G")
	set(CUDA_NVCC_FLAGS_RELEASE "-O3 --use_fast_math")
endif()

include_directories(include)

add_executable(matmul_test
		main.cu
		src/kernels/simple.cu
		src/kernels/shared_mem.cu
		src/kernels/vectorized.cu
		src/kernels/vectorized_opt.cu
		src/kernels/dispatch.cu
		src/kernels/high_shared.cu
		src/kernels/tensor_core.cu
		src/kernels/softmax.cu
		src/kernels/layernorm.cu
		src/kernels/batched.cu
		src/utils/timer.cu
		src/utils/validator.cu
)

target_link_libraries(matmul_test
		CUDA::cudart
		CUDA::cublas
)

set_target_properties(matmul_test PROPERTIES
		CUDA_RUNTIME_LIBRARY Shared
		CUDA_ARCHITECTURES 89
		CUDA_SEPARABLE_COMPILATION OFF
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

if(MSVC)
	target_compile_options(matmul_test PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/EHsc">
			$<$<COMPILE_LANGUAGE:CXX>:/EHsc>
	)

	target_compile_options(matmul_test PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/wd4067">
	)
endif()
