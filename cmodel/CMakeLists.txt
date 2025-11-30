cmake_minimum_required(VERSION 3.16)

# RSA C model test
set(RSA_TEST rsa_cmodel)
add_executable(${RSA_TEST} rsa_test.cpp rsa.cpp)
target_link_libraries(${RSA_TEST} GTest::gtest_main GTest::gtest)
add_test(${RSA_TEST} ${RSA_TEST})

# Tensor-Train Randomized SVD test
set(TT_RSVD_TEST tt_rsvd_test)
add_executable(${TT_RSVD_TEST} tt_rsvd_test.cpp tt_rsvd.cpp)
target_link_libraries(${TT_RSVD_TEST} GTest::gtest_main GTest::gtest)
add_test(${TT_RSVD_TEST} ${TT_RSVD_TEST})

# MATLAB comparison program
set(TT_MATLAB_COMPARE tt_matlab_compare)
add_executable(${TT_MATLAB_COMPARE} tt_matlab_compare.cpp tt_rsvd.cpp)
# No test registration - this is run manually after MATLAB generates test data

# Debug test program
set(TT_DEBUG_TEST tt_debug_test)
add_executable(${TT_DEBUG_TEST} tt_debug_test.cpp tt_rsvd.cpp)
# Simple debug test to check decomposition correctness
